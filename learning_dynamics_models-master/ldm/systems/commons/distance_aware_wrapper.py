"""
Distance-aware dynamics model wrapper.
=======================================
Wraps any dynamics model (e.g. SingleTrack + NeuralTireModel) and attenuates
the predicted state-derivatives when the current operating point is far from
the training distribution.

The proximity measure is the distance-to-dataset value returned by a small
neural network (``DistanceModel``) evaluated at the four-dimensional slip-
coordinate vector ``(sa_front, sr_front, sa_rear, sr_rear)``.

Attenuation rule
----------------
Let ``d = distance_model(slips)`` and ``τ = threshold``.

* If ``d ≤ τ``  →  scale = 1.0         (in-distribution, no dampening)
* If ``d > τ``  →  scale = τ / d        (proportional dampening)

This yields a smooth, monotonically-decreasing gain that equals 1 at the
threshold and approaches 0 as ``d → ∞``.

Usage
-----
::

    distance_model = DistanceModel.from_checkpoint("distance_model.pt")
    wrapped = DistanceAwareDynamicsWrapper(
        dynamics_model=single_track,
        distance_model=distance_model,
        threshold=0.5,
    )
    dx, tire_forces = wrapped(t, x, u)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.state_wrapper import StateWrapper
from ldm.systems.commons.distance_model import DistanceModel


class DistanceAwareDynamicsWrapper(nn.Module):
    """
    Wraps a dynamics model and dampens predicted accelerations outside the
    training distribution.

    Parameters
    ----------
    dynamics_model : nn.Module
        Any dynamics model whose ``forward(t, x, u)`` returns
        ``(state_derivatives, auxiliary)``.
        Must expose a ``vehicle_parameters`` attribute that is callable and
        returns a namedtuple with fields ``lf``, ``lr``, and ``eps``
        (i.e. a ``SingleTrackParams``-compatible object).
    distance_model : DistanceModel
        Trained distance-to-dataset network.
    threshold : float
        Distance value below which predictions are unmodified.  Above this
        value the derivatives are scaled by ``threshold / distance``.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        distance_model: DistanceModel,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.dynamics_model = dynamics_model
        self.distance_model = distance_model
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Slip coordinate computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slips(x_and_u: torch.Tensor, wp) -> torch.Tensor:
        """
        Compute the 4-D slip vector from a concatenated state+control tensor.

        Parameters
        ----------
        x_and_u : Tensor  (..., 6)
            ``[v_x, v_y, r, omega_wheels_rear, omega_wheels_front, delta]``
        wp : SingleTrackParams-compatible namedtuple

        Returns
        -------
        slips : Tensor  (..., 4)
            ``[sa_front, sr_front, sa_rear, sr_rear]``
        """
        wx = StateWrapper(x_and_u)
        sa_front = BaseTireModel.slip_angle_front_func(wx, wp)
        sr_front = BaseTireModel.slip_ratio_front_func(wx, wp)
        sa_rear  = BaseTireModel.slip_angle_rear_func(wx, wp)
        sr_rear  = BaseTireModel.slip_ratio_rear_func(wx, wp)
        return torch.stack([sa_front, sr_front, sa_rear, sr_rear], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, t, x, u):
        """
        Run the wrapped dynamics model and scale the output by the distance gate.

        Parameters
        ----------
        t : scalar or Tensor
        x : Tensor  (..., state_dim)
        u : Tensor  (..., control_dim)

        Returns
        -------
        dx_scaled : Tensor  (..., state_dim)
        auxiliary : whatever the wrapped model returns as its second value
        """
        # ── Call wrapped model ────────────────────────────────────────
        dx, aux = self.dynamics_model(t, x, u)

        # ── Compute slips ─────────────────────────────────────────────
        wp = self.dynamics_model.vehicle_parameters()
        x_and_u = torch.cat([x, u], dim=-1)
        slips = self._compute_slips(x_and_u, wp)

        # ── Evaluate distance ─────────────────────────────────────────
        with torch.no_grad():
            distance = self.distance_model(slips.float())   # (..., 1)

        # ── Compute scaling factor ────────────────────────────────────
        # scale = clamp(threshold / distance, max=1.0)
        # Safe against distance == 0 (shouldn't occur in practice but just in case)
        scale = torch.clamp(
            self.threshold / (distance + 1e-12),
            max=1.0,
        )                                                     # (..., 1)

        # ── Apply scaling ─────────────────────────────────────────────
        dx_scaled = dx * scale

        return dx_scaled, aux

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------

    def state_weights(self):
        return self.dynamics_model.state_weights()

    def get_state_names(self):
        return self.dynamics_model.get_state_names()

    def get_control_names(self):
        return self.dynamics_model.get_control_names()

    def get_parameters(self):
        return self.dynamics_model.get_parameters()

    def get_parameters_vector(self):
        return self.dynamics_model.get_parameters_vector()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_scale(self, t, x, u) -> torch.Tensor:
        """
        Return the scaling factor without running the dynamics model.
        Useful for diagnostics and visualisation.

        Returns
        -------
        scale : Tensor  (..., 1)
        """
        wp = self.dynamics_model.vehicle_parameters()
        x_and_u = torch.cat([x, u], dim=-1)
        slips = self._compute_slips(x_and_u, wp)
        with torch.no_grad():
            distance = self.distance_model(slips.float())
        scale = torch.clamp(self.threshold / (distance + 1e-12), max=1.0)
        return scale

    def get_distance(self, t, x, u) -> torch.Tensor:
        """
        Return the raw predicted distance without running the dynamics model.

        Returns
        -------
        distance : Tensor  (..., 1)
        """
        wp = self.dynamics_model.vehicle_parameters()
        x_and_u = torch.cat([x, u], dim=-1)
        slips = self._compute_slips(x_and_u, wp)
        with torch.no_grad():
            distance = self.distance_model(slips.float())
        return distance
