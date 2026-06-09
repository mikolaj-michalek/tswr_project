"""
Combined base + residual dynamics model.

The combined model has **exactly the same call signature** as the nominal base
model::

    dx, tire_forces, slips = combined_model(t, x, u)

so it is a drop-in replacement inside ``RolloutModel`` / ``RolloutModelWithHistory``
and inside any RL environment.

The only additional API is ``set_z`` which allows the caller to fix the latent
code before a rollout::

    # Broadcast one z to every element in the batch
    combined_model.set_z(torch.zeros(z_dim))

    # Different z per batch element (B must match the batch at call time)
    combined_model.set_z(torch.randn(B, z_dim))

    # Reset to zeros
    combined_model.set_z(None)

At call time the stored z is broadcast to the current batch dimension, so the
batch size does not need to be known when ``set_z`` is called.

Architecture
------------
    dx  =  base_model(t, x, u)          [frozen by default]
         + residual_mlp(t, x, u, z)     [learned correction]
"""

import torch
import torch.nn as nn
from typing import Optional

from ldm.systems.mlp.residual_mlp import ResidualMlp


class CombinedResidualDynamicsModel(nn.Module):
    """
    Drop-in replacement for the base dynamics model, augmented with a
    VAE-conditioned residual MLP.

    Parameters
    ----------
    base_model : nn.Module
        Trained base dynamics model.  Expected call signature::
            dx, tire_forces, slips = base_model(t, x, u)
    residual_mlp : ResidualMlp
        Trained residual MLP.
    z_dim : int
        Dimensionality of the latent code.
    freeze_base : bool
        If True (default), the base model parameters are frozen (no gradients).
    """

    def __init__(
        self,
        base_model: nn.Module,
        residual_mlp: ResidualMlp,
        z_dim: int,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model   = base_model
        self.residual_mlp = residual_mlp
        self.z_dim        = z_dim

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)

        # z is stored as a plain tensor attribute (not a parameter or buffer)
        # so it doesn't appear in state_dict / optimizer, but is moved by .to()
        # via the custom .to() override below.
        self._z: Optional[torch.Tensor] = None  # (z_dim,) or (B, z_dim) or None

    # ------------------------------------------------------------------
    # z management
    # ------------------------------------------------------------------

    def set_z(self, z: Optional[torch.Tensor]) -> "CombinedResidualDynamicsModel":
        """
        Set the latent code used for the next rollout(s).

        Parameters
        ----------
        z : Tensor of shape ``(z_dim,)`` or ``(B, z_dim)``, or ``None``
            - ``None`` → use zeros (same as ``torch.zeros(z_dim)``).
            - 1-D tensor of length ``z_dim`` → broadcast to every batch element.
            - 2-D tensor of shape ``(B, z_dim)`` → one code per batch element.
        """
        if z is None:
            self._z = None
        else:
            if z.dim() not in (1, 2):
                raise ValueError(
                    f"z must be 1-D (z_dim,) or 2-D (B, z_dim), got shape {tuple(z.shape)}"
                )
            if z.dim() == 1 and z.shape[0] != self.z_dim:
                raise ValueError(
                    f"1-D z must have length z_dim={self.z_dim}, got {z.shape[0]}"
                )
            if z.dim() == 2 and z.shape[1] != self.z_dim:
                raise ValueError(
                    f"2-D z must have shape (B, z_dim={self.z_dim}), got {tuple(z.shape)}"
                )
            self._z = z
        return self  # allow chaining

    def get_z(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Return the latent code expanded to ``(batch_size, z_dim)``.

        If no z has been set, returns zeros.
        """
        if self._z is None:
            return torch.zeros(batch_size, self.z_dim, device=device)

        z = self._z.to(device)
        if z.dim() == 1:
            # (z_dim,) → (B, z_dim)
            return z.unsqueeze(0).expand(batch_size, -1)
        else:
            # (B, z_dim) — verify batch size matches
            if z.shape[0] != batch_size:
                raise RuntimeError(
                    f"Stored z has batch size {z.shape[0]} but model was called "
                    f"with batch size {batch_size}.  Call set_z() again with the "
                    f"correct batch size or use a 1-D z for broadcasting."
                )
            return z

    # ------------------------------------------------------------------
    # nn.Module device / dtype movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self._z is not None:
            self._z = self._z.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        if self._z is not None:
            self._z = self._z.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        if self._z is not None:
            self._z = self._z.cpu()
        return self

    # ------------------------------------------------------------------
    # Forward — same interface as the base model
    # ------------------------------------------------------------------

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
    ):
        """
        Compute the combined state derivative.

        Parameters
        ----------
        t : Tensor  ``(1,)``            — current time
        x : Tensor  ``(B, state_dim)``  — current state
        u : Tensor  ``(B, control_dim)``— current control

        Returns
        -------
        dx          : Tensor ``(B, state_dim)`` — combined state derivative
        tire_forces : forwarded from base model unchanged
        slips       : forwarded from base model unchanged
        """
        B = x.shape[0]

        # Base model (frozen)
        with torch.no_grad():
            dx_base, tire_forces, slips = self.base_model(t, x, u)
        # Pad to full state_dim — mirrors what RolloutModel._euler_step does so
        # the combined model is a safe drop-in for any base model output size.
        dx_base = torch.nn.functional.pad(
            dx_base, (0, x.shape[-1] - dx_base.shape[-1])
        )

        # Residual correction
        z   = self.get_z(B, x.device)        # (B, z_dim)
        dR  = self.residual_mlp(t, x, u, z)  # (B, state_dim)

        return dx_base + dR, tire_forces, slips
