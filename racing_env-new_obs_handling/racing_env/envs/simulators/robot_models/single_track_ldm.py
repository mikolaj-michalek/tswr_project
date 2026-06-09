"""SingleTrackLDMPacejkaModel — a drop-in replacement for SingleTrackPacejkaModel
that embeds its own parameters loaded from an exported LDM YAML config.

The class is fully compatible with the Simulator's call signature
    model(t, x, p_vehicle, p_tire_front, p_tire_rear)
while also working standalone (pass None for any param tensor to fall back to the
embedded buffers loaded from the YAML).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml

from racing_env.envs.simulators.robot_models.base_tire_model import BaseTireModel
from racing_env.envs.simulators.robot_models.pacejka_params import PacejkaParameters
from racing_env.envs.simulators.robot_models.pacejka_tire_model import PacejkaTireModel
from racing_env.envs.simulators.robot_models.single_track_params import VehicleParameters
from racing_env.utils.state_wrapper import StateWrapper

# Canonical key order that mirrors xray_new.yaml → must match VehicleParameters indices
_VEHICLE_KEYS = [
    "m", "g", "I_z", "L", "lr",
    "Cd0", "Cd1", "Cd2", "mu_static",
    "I_e", "K_fi", "b0", "b1", "R",
    "tau_omega", "tau_delta",
]
_TIRE_KEYS = ["Sx_p", "Alpha_p", "By", "Cy", "Dy", "Ey", "Bx", "Cx", "Dx", "Ex"]


def _yaml_to_tensor(section: dict, keys: list[str]) -> torch.Tensor:
    return torch.tensor([[section[k] for k in keys]], dtype=torch.float32)


class SingleTrackLDMPacejkaModel(torch.nn.Module):
    """Single-track Pacejka model pre-loaded with parameters from an LDM-exported YAML.

    Parameters come from a YAML config in ``xray_new.yaml`` format (produced by
    ``ldm.export.export_to_racing_env.export_ldm_to_yaml``).  When used inside
    the Simulator the external param tensors (p_vehicle, p_tire_front, p_tire_rear)
    take priority; pass ``None`` for standalone / test usage.

    Args:
        yaml_path: Path to the vehicle YAML config (xray_new.yaml format).
    """

    def __init__(self, yaml_path: str) -> None:
        super().__init__()
        self.eps = 1e-6

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        self.register_buffer(
            "_vehicle_params", _yaml_to_tensor(cfg["Vehicle"], _VEHICLE_KEYS)
        )
        self.register_buffer(
            "_tire_front_params", _yaml_to_tensor(cfg["TireFront"], _TIRE_KEYS)
        )
        self.register_buffer(
            "_tire_rear_params", _yaml_to_tensor(cfg["TireRear"], _TIRE_KEYS)
        )

        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()
        self.tire_model = PacejkaTireModel()

    # ------------------------------------------------------------------
    # Helpers to retrieve embedded default params (useful for simulator
    # initialisation or standalone evaluation)
    # ------------------------------------------------------------------

    def get_default_vehicle_params(self, batch_size: int = 1) -> torch.Tensor:
        return self._vehicle_params.expand(batch_size, -1)

    def get_default_tire_front_params(self, batch_size: int = 1) -> torch.Tensor:
        return self._tire_front_params.expand(batch_size, -1)

    def get_default_tire_rear_params(self, batch_size: int = 1) -> torch.Tensor:
        return self._tire_rear_params.expand(batch_size, -1)

    # ------------------------------------------------------------------
    # Forward — identical dynamics to SingleTrackPacejkaModel
    # ------------------------------------------------------------------

    def forward(
        self,
        t: float,
        x: torch.Tensor,
        p_vehicle: Optional[torch.Tensor] = None,
        p_tire_front: Optional[torch.Tensor] = None,
        p_tire_rear: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute state derivatives.

        Args:
            t: Current time (unused in this model).
            x: State tensor of shape ``[batch, state_dim]`` or ``[state_dim]``.
            p_vehicle: Vehicle parameter tensor ``[batch, 16]``; uses embedded
                defaults when ``None``.
            p_tire_front: Front-tyre Pacejka params ``[batch, 10]``; uses
                embedded defaults when ``None``.
            p_tire_rear: Rear-tyre Pacejka params ``[batch, 10]``; uses
                embedded defaults when ``None``.

        Returns:
            State-derivative tensor with the same shape as ``x``.
        """
        batch = x.shape[0] if x.dim() > 1 else 1

        p_v  = p_vehicle    if p_vehicle    is not None else self._vehicle_params.expand(batch, -1)
        p_tf = p_tire_front if p_tire_front is not None else self._tire_front_params.expand(batch, -1)
        p_tr = p_tire_rear  if p_tire_rear  is not None else self._tire_rear_params.expand(batch, -1)

        p   = self.vehicle_parameters(p_v)
        wx  = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tf)
        wp_tire_r = self.tire_model_parameters(p_tr)

        alpha_f     = BaseTireModel.slip_angle_front_func(wx, p)
        alpha_r     = BaseTireModel.slip_angle_rear_func(wx, p)
        slip_ratio_f = BaseTireModel.slip_ratio_front_func(wx, p)
        slip_ratio_r = BaseTireModel.slip_ratio_func(wx, p)

        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r)
        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        Fy_f, Fx_f = Fy_f * wx.front_friction, Fx_f * wx.front_friction
        Fy_r, Fx_r = Fy_r * wx.rear_friction,  Fx_r * wx.rear_friction

        self.last_tire_dynamics = {
            "Fx_f": Fx_f, "Fx_r": Fx_r, "Fy_f": Fy_f, "Fy_r": Fy_r,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "slip_ratio_f": slip_ratio_f, "slip_ratio_r": slip_ratio_r,
        }

        F_drag = (
            p.Cd0 * torch.sign(wx.v_x)
            + p.Cd1 * wx.v_x
            + p.Cd2 * wx.v_x * wx.v_x
        )

        v_x_dot = (1.0 / p.m) * (
            Fx_r + Fx_f * torch.cos(wx.delta) - Fy_f * torch.sin(wx.delta)
            - F_drag + p.m * wx.v_y * wx.r
        )
        v_y_dot = (1.0 / p.m) * (
            Fx_f * torch.sin(wx.delta) + Fy_r + Fy_f * torch.cos(wx.delta)
            - p.m * wx.v_x * wx.r
        )
        r_dot = (1.0 / p.I_z) * (
            (Fx_f * torch.sin(wx.delta) + Fy_f * torch.cos(wx.delta)) * p.lf
            - Fy_r * p.lr
        )

        omega_wheels_dot = (wx.omega_wheels_ref - wx.omega_wheels) / p.tau_omega
        delta_dot        = (wx.delta_ref - wx.delta) / p.tau_delta

        x_dot   = wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw)
        y_dot   = wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw)
        yaw_dot = wx.r

        zeros_ff = torch.zeros_like(wx.front_friction)
        zeros_rf = torch.zeros_like(wx.rear_friction)
        zeros_dr = torch.zeros_like(wx.delta_ref)
        zeros_ow = torch.zeros_like(wx.omega_wheels_ref_dot)

        if x.dim() == 1:
            return torch.tensor([
                x_dot, y_dot, yaw_dot,
                v_x_dot, v_y_dot, r_dot,
                omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot,
                zeros_ff, zeros_rf, zeros_dr, zeros_ow,
            ])
        return torch.stack([
            x_dot, y_dot, yaw_dot,
            v_x_dot, v_y_dot, r_dot,
            omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot,
            zeros_ff, zeros_rf, zeros_dr, zeros_ow,
        ], dim=1)
