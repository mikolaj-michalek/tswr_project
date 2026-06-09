"""
Residual dynamics MLP conditioned on a latent code z.

Classes
-------
ResidualPreprocessor
    Normalises the [state | control] portion of the input via
    CarObservationPreprocessor and passes z through unchanged.

ResidualMlp
    Wraps the base Mlp so it can be called as
        forward(t, x, u, z) -> dR : (B, state_dim)
    The output represents a state-space derivative; multiply by Tp to get
    the one-step residual correction.
"""

import torch
import torch.nn as nn

from ldm.systems.mlp.mlp import Mlp
from ldm.systems.f1tenth.utils.f1tenth_observation_preprocesor import (
    CarObservationPreprocessor,
    STATE_DIM as _STATE_DIM,
    CONTROL_DIM as _CONTROL_DIM,
)


class ResidualPreprocessor(nn.Module):
    """
    Preprocessor for the residual MLP input vector [state | control | z].

    The [state | control] slice is normalised by CarObservationPreprocessor;
    the z slice is passed through unchanged.

    Parameters
    ----------
    state_dim   : dimensionality of the state  (default: 5 for F1tenth)
    control_dim : dimensionality of the control (default: 3 for F1tenth)
    """

    def __init__(
        self,
        state_dim:   int = _STATE_DIM,
        control_dim: int = _CONTROL_DIM,
    ):
        super().__init__()
        self.obs_prep    = CarObservationPreprocessor()
        self.xu_dim      = state_dim + control_dim

    def forward(self, xu_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xu_z : (..., state_dim + control_dim + z_dim)
        Returns:
            (..., state_dim + control_dim + z_dim)  — xu part normalised
        """
        xu = xu_z[..., : self.xu_dim]
        z  = xu_z[..., self.xu_dim :]
        return torch.cat([self.obs_prep(xu), z], dim=-1)


class ResidualMlp(nn.Module):
    """
    Residual dynamics MLP conditioned on a latent code z.

    Wraps the base :class:`Mlp` (which expects ``forward(t, x, u)``) so it
    can be called with an explicit z argument.  Internally, (x, u, z) are
    concatenated into a single vector and fed to the inner Mlp as ``x``
    with an empty ``u``.

    Parameters
    ----------
    state_dim   : output / input state dimensionality
    control_dim : control input dimensionality
    z_dim       : latent code dimensionality
    hidden_sizes: list of hidden layer widths, e.g. [128, 128]
    activation  : activation module (default: ReLU)
    compile     : whether to torch.compile the inner Mlp

    Forward signature
    -----------------
    forward(t, x, u, z) -> dR

        t : (1,)            – current time (passed to Mlp, usually unused)
        x : (B, state_dim)
        u : (B, control_dim)
        z : (B, z_dim)
        dR: (B, state_dim)  – predicted residual derivative
                              multiply by Tp to get one-step residual
    """

    def __init__(
        self,
        state_dim:    int,
        control_dim:  int,
        z_dim:        int,
        hidden_sizes: list,
        activation:   nn.Module = None,
        compile:      bool = False,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        mlp_input_dim = state_dim + control_dim - 2 + z_dim
        layer_sizes   = [mlp_input_dim] + list(hidden_sizes) + [state_dim]

        self.mlp = Mlp(
            preprocessor = ResidualPreprocessor(
                state_dim=state_dim, control_dim=control_dim
            ),
            layer_sizes  = layer_sizes,
            activation   = activation,
            compile      = compile,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            t : (1,)
            x : (B, state_dim)
            u : (B, control_dim)
            z : (B, z_dim)
        Returns:
            dR : (B, state_dim)
        """
        xu_z = torch.cat([x, u, z], dim=-1)
        # Pass full vector as x; inner Mlp will cat([x, u]) = cat([xu_z, []])
        dR, _, _ = self.mlp(t, xu_z, torch.empty(x.shape[0], 0, device=x.device))
        return dR
