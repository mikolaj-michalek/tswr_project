"""
1-D CNN future-trajectory encoders for residual-dynamics conditioning.

Architecture
------------
Input  : (batch, horizon, state_dim)   – future ground-truth state trajectory
         Transposed internally to Conv1d layout:
         (batch, state_dim, horizon)

Backbone: stacked Conv1d → BatchNorm1d → ReLU blocks
Neck    : global average pooling → flatten
Head    : depends on the chosen encoder type

Implemented variants
--------------------
- ``CnnVAEFutureEncoder``: CVAE posterior with ``mu`` / ``log_var`` heads and
  the usual reparameterisation trick.
- ``CnnBoundedFutureEncoder``: classical deterministic conditioned autoencoder
  with a ``tanh`` latent head, giving ``z ∈ [-1, 1]^N``.
"""

from typing import List
import torch
import torch.nn as nn


def _build_conv_backbone(in_channels: int, channels: List[int], kernel_size: int):
    conv_layers = []
    in_ch = in_channels
    for out_ch in channels:
        conv_layers += [
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        in_ch = out_ch
    return nn.Sequential(*conv_layers), in_ch


class CnnVAEFutureEncoder(nn.Module):
    """
    CNN VAE encoder that conditions on the future state trajectory.

    Parameters
    ----------
    state_dim   : dimensionality of each state frame
    z_dim       : dimensionality of the latent code z
    horizon     : number of future time steps in the trajectory
    channels    : list of out-channel sizes for successive Conv1d layers
                  e.g. [32, 64, 64]
    kernel_size : convolutional kernel size (same for all layers)
    log_var_clip: clamp log_var to [-log_var_clip, log_var_clip] for stability
    extra_dim   : number of additional per-step features concatenated to the
                  state along the feature axis before the CNN backbone.
                  Typical use-cases:
                  - conditioning on controls  → extra_dim = ctrl_dim
                  - conditioning on nominal rollout → extra_dim = state_dim
                  - both                       → extra_dim = state_dim + ctrl_dim
                  The caller is responsible for concatenating those features
                  into a single ``(B, H, state_dim + extra_dim)`` tensor and
                  passing it to ``forward``.
    """

    def __init__(
        self,
        state_dim: int,
        z_dim: int,
        horizon: int,
        channels: List[int] = (32, 64, 64),
        kernel_size: int = 5,
        log_var_clip: float = 4.0,
        extra_dim: int = 0,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.log_var_clip = log_var_clip
        self.uses_kl_regularization = True

        self.conv_backbone, in_ch = _build_conv_backbone(
            in_channels=state_dim + extra_dim,
            channels=channels,
            kernel_size=kernel_size,
        )

        # Global average pool → (B, last_out_ch)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Projection heads for mu and log_var
        self.fc_mu      = nn.Linear(in_ch, z_dim)
        self.fc_log_var = nn.Linear(in_ch, z_dim)

    # ------------------------------------------------------------------

    def encode(self, X_future: torch.Tensor):
        """
        Args:
            X_future : (B, H, state_dim + extra_dim)  — future features
        Returns:
            mu      : (B, z_dim)
            log_var : (B, z_dim)
        """
        # Conv1d expects (B, C, L)
        x = X_future.transpose(1, 2)          # (B, state_dim, H)
        x = self.conv_backbone(x)             # (B, channels[-1], H)
        x = self.gap(x).squeeze(-1)           # (B, channels[-1])
        mu = self.fc_mu(x)                    # (B, z_dim)
        log_var = self.fc_log_var(x).clamp(-self.log_var_clip, self.log_var_clip)
        return mu, log_var

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample z = mu + eps * std  where eps ~ N(0, I).
        During eval (no_grad) the mean is returned deterministically.
        """
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, X_future: torch.Tensor):
        """
        Args:
            X_future : (B, H, state_dim + extra_dim)  — future features
        Returns:
            z       : (B, z_dim)   – sampled latent code
            mu      : (B, z_dim)
            log_var : (B, z_dim)
        """
        mu, log_var = self.encode(X_future)
        z = self.reparameterise(mu, log_var)
        return z, mu, log_var

    # ------------------------------------------------------------------
    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Analytical KL( q(z|x) || N(0,I) ) averaged over the batch.
        Returns a scalar tensor.
        """
        return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()


class CnnBoundedFutureEncoder(nn.Module):
    """
    Deterministic CNN encoder with a bounded latent code ``z ∈ [-1, 1]^N``.

    This is a classical conditioned autoencoder: the future trajectory encoder
    maps the conditioning sequence directly to a latent code via a ``tanh``
    head and does not use KL regularisation.
    """

    def __init__(
        self,
        state_dim: int,
        z_dim: int,
        horizon: int,
        channels: List[int] = (32, 64, 64),
        kernel_size: int = 5,
        extra_dim: int = 0,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.horizon = horizon
        self.uses_kl_regularization = False

        self.conv_backbone, in_ch = _build_conv_backbone(
            in_channels=state_dim + extra_dim,
            channels=channels,
            kernel_size=kernel_size,
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Linear(in_ch, z_dim)
        self.latent_activation = nn.Tanh()

    def encode(self, X_future: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X_future : (B, H, state_dim + extra_dim)  — future features
        Returns:
            z : (B, z_dim) bounded latent code in ``[-1, 1]``
        """
        x = X_future.transpose(1, 2)
        x = self.conv_backbone(x)
        x = self.gap(x).squeeze(-1)
        return self.latent_activation(self.fc_latent(x))

    def forward(self, X_future: torch.Tensor):
        """
        Returns:
            z, None, None
        """
        z = self.encode(X_future)
        return z, None, None
