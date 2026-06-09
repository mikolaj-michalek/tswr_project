"""
Distance-to-dataset neural network model.
==========================================
Wraps the MLP trained by ``experiments/train_distance_model.py`` and
exposes a simple ``predict(slips)`` interface that returns the predicted
distance value in the original (un-normalised) space.

The checkpoint stores:
    - ``model_state``  : state_dict of the DistanceModel, including the MLP
                         weights *and* the normalisation buffers
                         (``x_mean``, ``x_std``, ``y_mean``, ``y_std``).
    - ``hidden_sizes`` : list[int]  – needed to reconstruct the architecture.
    - ``activation``   : str  ("relu" | "tanh" | "gelu" | "silu")

    Legacy checkpoints (before this change) stored the normalisation stats
    as top-level keys and ``model_state`` contained only the MLP weights
    (with a ``"net."`` prefix).  ``from_checkpoint`` handles both formats.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ── Architecture (mirrors train_distance_model.py) ────────────────────────────

def _build_mlp(hidden_sizes: List[int], activation: str) -> nn.Sequential:
    act_cls = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }[activation]
    layers: list[nn.Module] = []
    in_dim = 4
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), act_cls()]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


class DistanceModel(nn.Module):
    """
    Distance-to-dataset MLP with built-in I/O normalisation.

    The network is built internally from ``hidden_sizes`` and ``activation``;
    normalisation statistics are stored as non-trainable buffers so they travel
    with the model across devices and are saved/restored via ``state_dict``.

    Parameters
    ----------
    hidden_sizes : list[int]
        Width of each hidden layer.
    activation : str
        Activation function key: ``"relu"``, ``"tanh"``, ``"gelu"``, or
        ``"silu"``.
    x_mean, x_std : Tensor of shape (4,)
        Input normalisation statistics.
    y_mean, y_std : Tensor of shape (1,)
        Output normalisation statistics.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        activation: str,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.net = _build_mlp(hidden_sizes, activation)
        # Register as buffers so they move with .to(device) and are not
        # treated as trainable parameters.
        self.register_buffer("x_mean", x_mean.float())
        self.register_buffer("x_std", x_std.float())
        self.register_buffer("y_mean", y_mean.float())
        self.register_buffer("y_std", y_std.float())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str,
                        map_location: str = "cpu") -> "DistanceModel":
        """Load a checkpoint produced by ``train_distance_model.py``."""
        ckpt = torch.load(checkpoint_path, map_location=map_location,
                          weights_only=False)
        model = cls(
            hidden_sizes=ckpt["hidden_sizes"],
            activation=ckpt["activation"],
            x_mean=torch.zeros(4),
            x_std=torch.ones(4),
            y_mean=torch.zeros(1),
            y_std=torch.ones(1),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, slips: torch.Tensor) -> torch.Tensor:
        """
        Predict distance to dataset.

        Parameters
        ----------
        slips : Tensor  (..., 4)
            Slip coordinates ordered as
            ``[sa_front, sr_front, sa_rear, sr_rear]``.

        Returns
        -------
        distance : Tensor  (..., 1)
            Predicted distance in the original (un-normalised) space.
        """
        x_n = (slips - self.x_mean) / self.x_std
        y_n = self.net(x_n)
        return y_n * self.y_std + self.y_mean
