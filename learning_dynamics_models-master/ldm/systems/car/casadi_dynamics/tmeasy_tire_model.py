import numpy as np
import casadi as ca
import torch

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.dynamics.tmeasy_params import TMeasyParameters
from ldm.systems.car.dynamics.tmeasy_tire_model import TMeasyTireModel


class CasadiTMeasyTireModel(CasadiBaseTireModel):
    def __init__(self):
        super().__init__()
        self.front_tire_model_parameters = TMeasyParameters()
        self.rear_tire_model_parameters = TMeasyParameters()

    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        """
        Implements the TMeasy tire model for combined slip (CasADi version).
        wp is a dictionary containing TMeasy params:
            df_x0, s_mx, mu_mx, s_gx, mu_gx  (longitudinal)
            df_y0, s_my, mu_my, s_gy, mu_gy  (lateral)
        Returns: mu_x, mu_y (normalized forces, i.e. friction coefficients)
        """
        # 1. Coordinate Transforms
        sx = slip_ratio
        sy = ca.tan(slip_angle_rad)

        # 2. Normalization Factors (Effective Slip at Peak)
        s_nx = sx / (wp['s_mx'] + 1e-8)
        s_ny = sy / (wp['s_my'] + 1e-8)

        # 3. Generalized Slip (Normalized Combined Slip)
        s_norm = ca.sqrt(s_nx**2 + s_ny**2) + 1e-8

        # 4. Angle functions for interpolation
        cos_phi = s_nx / s_norm
        sin_phi = s_ny / s_norm

        # 5. Interpolate Parameters for Combined Curve
        mu_M = ca.sqrt((wp['mu_mx'] * cos_phi)**2 + (wp['mu_my'] * sin_phi)**2)
        mu_G = ca.sqrt((wp['mu_gx'] * cos_phi)**2 + (wp['mu_gy'] * sin_phi)**2)

        df0_norm = ca.sqrt((wp['df_x0'] * wp['s_mx'] * cos_phi)**2 +
                           (wp['df_y0'] * wp['s_my'] * sin_phi)**2)

        s_gx_norm = wp['s_gx'] / wp['s_mx']
        s_gy_norm = wp['s_gy'] / wp['s_my']
        s_G_norm = ca.sqrt((s_gx_norm * cos_phi)**2 + (s_gy_norm * sin_phi)**2)

        # 6. Calculate Resultant Force F(s)

        # --- Region 1: Adhesion (s_norm <= 1.0) ---
        # Rational function: F = (df0 * s) / (1 + b*s + s^2)
        # Conditions: F(1) = mu_M, F'(1) = 0  =>  b = df0/mu_M - 2, c = 1
        b = (df0_norm / (mu_M + 1e-8)) - 2.0
        b = ca.fmax(b, 0.0)  # clamp to non-negative for physical validity

        F_adhesion = (df0_norm * s_norm) / (1.0 + b * s_norm + s_norm**2 + 1e-8)

        # --- Region 2: Sliding transition (1.0 < s_norm <= s_G_norm) ---
        # Hermite cubic blend from mu_M (at s=1) to mu_G (at s=s_G_norm)
        t = (s_norm - 1.0) / (s_G_norm - 1.0 + 1e-8)
        t = ca.fmin(ca.fmax(t, 0.0), 1.0)  # clamp to [0, 1]
        blend = 3.0 * t**2 - 2.0 * t**3
        F_sliding_trans = mu_M + (mu_G - mu_M) * blend

        # --- Region 3: Full Sliding (s_norm > s_G_norm) ---
        F_full_sliding = mu_G

        # Combine Regions using nested if_else
        mu_total = ca.if_else(
            s_norm <= 1.0,
            F_adhesion,
            ca.if_else(
                s_norm <= s_G_norm,
                F_sliding_trans,
                F_full_sliding
            )
        )

        # 7. Project back to X and Y
        mu_x = mu_total * cos_phi
        mu_y = mu_total * sin_phi

        return mu_x, -mu_y


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    torch_model = TMeasyTireModel()
    casadi_model = CasadiTMeasyTireModel()

    # Use exactly the same trained parameters for both implementations.
    casadi_model.front_tire_model_parameters = torch_model.front_tire_model_parameters

    wp_torch = torch_model.front_tire_model_parameters()
    wp_casadi = torch_model.front_tire_model_parameters.get_parameters_dict()

    sa_sym = ca.MX.sym('sa')
    sr_sym = ca.MX.sym('sr')
    fx_sym, fy_sym = casadi_model.tire_forces_model(sa_sym, sr_sym, wp_casadi)
    tire_force_func = ca.Function('tmeasy_tire_force', [sa_sym, sr_sym], [fx_sym, fy_sym])

    num_tests = 200
    slip_angles = np.random.uniform(-0.8, 0.8, size=num_tests)
    slip_ratios = np.random.uniform(-1.0, 1.0, size=num_tests)

    fx_abs_err_max = 0.0
    fy_abs_err_max = 0.0

    for slip_angle_rad, slip_ratio in zip(slip_angles, slip_ratios):
        fx_torch, fy_torch = torch_model.tire_forces_model(
            torch.tensor(slip_angle_rad, dtype=torch.float32),
            torch.tensor(slip_ratio, dtype=torch.float32),
            wp_torch,
        )

        fx_casadi, fy_casadi = tire_force_func(float(slip_angle_rad), float(slip_ratio))

        fx_torch_val = float(fx_torch.item())
        fy_torch_val = float(fy_torch.item())
        fx_casadi_val = float(fx_casadi)
        fy_casadi_val = float(fy_casadi)

        fx_abs_err = abs(fx_torch_val - fx_casadi_val)
        fy_abs_err = abs(fy_torch_val - fy_casadi_val)

        fx_abs_err_max = max(fx_abs_err_max, fx_abs_err)
        fy_abs_err_max = max(fy_abs_err_max, fy_abs_err)

    atol = 4e-5
    rtol = 2e-5

    assert np.isclose(fx_abs_err_max, 0.0, atol=atol, rtol=rtol), (
        f"Fx mismatch too large: max_abs_err={fx_abs_err_max:.3e}, atol={atol}, rtol={rtol}"
    )
    assert np.isclose(fy_abs_err_max, 0.0, atol=atol, rtol=rtol), (
        f"Fy mismatch too large: max_abs_err={fy_abs_err_max:.3e}, atol={atol}, rtol={rtol}"
    )

    print("CasADi vs Torch TMeasy numerical check passed.")
    print(f"max |Fx_torch - Fx_casadi| = {fx_abs_err_max:.3e}")
    print(f"max |Fy_torch - Fy_casadi| = {fy_abs_err_max:.3e}")
