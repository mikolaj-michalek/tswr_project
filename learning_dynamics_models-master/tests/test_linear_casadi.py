import torch
import numpy as np
import casadi as ca
import unittest

from ldm.systems.car.casadi_dynamics.linear_tire_model import CasadiLinearTireModel
from ldm.systems.car.dynamics.linear_tire_model import LinearTireModel
from ldm.systems.car.dynamics.linear_params import LinearParameters


class TestLinearTireModelEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tire_torch = LinearTireModel()
        self.tire_casadi = CasadiLinearTireModel()

        # Sync parameters
        self.tire_casadi.front_tire_model_parameters.load_state_dict(
            self.tire_torch.front_tire_model_parameters.state_dict()
        )
        self.tire_casadi.rear_tire_model_parameters.load_state_dict(
            self.tire_torch.rear_tire_model_parameters.state_dict()
        )

    def _get_tire_params_dict(self, params_module):
        p = params_module()
        return {k: getattr(p, k).detach().item() for k in p._fields}

    def _compare_tire_forces(self, slip_angle_val, slip_ratio_val, label=""):
        # --- Torch ---
        sa_t = torch.tensor([[slip_angle_val]], dtype=torch.float32)
        sr_t = torch.tensor([[slip_ratio_val]], dtype=torch.float32)
        wp_torch = self.tire_torch.front_tire_model_parameters()

        fx_t, fy_t = LinearTireModel.tire_forces_model(sa_t, sr_t, wp_torch)
        fx_t_np = fx_t.detach().numpy().flatten()
        fy_t_np = fy_t.detach().numpy().flatten()

        # --- CasADi ---
        sa_sym = ca.MX.sym('sa')
        sr_sym = ca.MX.sym('sr')
        wp_casadi = self._get_tire_params_dict(self.tire_casadi.front_tire_model_parameters)

        fx_sym, fy_sym = self.tire_casadi.tire_forces_model(sa_sym, sr_sym, wp_casadi)
        f_casadi = ca.Function('f', [sa_sym, sr_sym], [fx_sym, fy_sym])
        res = f_casadi(slip_angle_val, slip_ratio_val)

        fx_c_np = np.array(res[0]).flatten()
        fy_c_np = np.array(res[1]).flatten()

        np.testing.assert_allclose(fx_t_np, fx_c_np, rtol=1e-5, atol=1e-7,
                                   err_msg=f"Fx mismatch {label}")
        np.testing.assert_allclose(fy_t_np, fy_c_np, rtol=1e-5, atol=1e-7,
                                   err_msg=f"Fy mismatch {label}")

    def test_nominal(self):
        self._compare_tire_forces(0.05, 0.03, "nominal")

    def test_zero_inputs(self):
        self._compare_tire_forces(0.0, 0.0, "zero inputs")

    def test_high_slip_angle(self):
        self._compare_tire_forces(0.3, 0.01, "high slip angle")

    def test_high_slip_ratio(self):
        self._compare_tire_forces(0.01, 0.3, "high slip ratio")

    def test_negative_slip_angle(self):
        self._compare_tire_forces(-0.1, 0.05, "negative sa")

    def test_negative_slip_ratio(self):
        self._compare_tire_forces(0.1, -0.05, "negative sr")

    def test_saturation(self):
        """Large slips that exceed friction circle → saturation branch."""
        self._compare_tire_forces(0.5, 0.5, "saturation")

    def test_linear_region(self):
        """Very small slips → below friction circle."""
        self._compare_tire_forces(0.001, 0.001, "linear region")

    def test_rear_tire_params(self):
        slip_angle_val, slip_ratio_val = 0.08, 0.06

        sa_t = torch.tensor([[slip_angle_val]], dtype=torch.float32)
        sr_t = torch.tensor([[slip_ratio_val]], dtype=torch.float32)
        wp_torch = self.tire_torch.rear_tire_model_parameters()

        fx_t, fy_t = LinearTireModel.tire_forces_model(sa_t, sr_t, wp_torch)

        sa_sym = ca.MX.sym('sa')
        sr_sym = ca.MX.sym('sr')
        wp_casadi = self._get_tire_params_dict(self.tire_casadi.rear_tire_model_parameters)

        fx_sym, fy_sym = self.tire_casadi.tire_forces_model(sa_sym, sr_sym, wp_casadi)
        f_casadi = ca.Function('f', [sa_sym, sr_sym], [fx_sym, fy_sym])
        res = f_casadi(slip_angle_val, slip_ratio_val)

        np.testing.assert_allclose(fx_t.detach().numpy().flatten(),
                                   np.array(res[0]).flatten(),
                                   rtol=1e-5, atol=1e-7, err_msg="Rear Fx mismatch")
        np.testing.assert_allclose(fy_t.detach().numpy().flatten(),
                                   np.array(res[1]).flatten(),
                                   rtol=1e-5, atol=1e-7, err_msg="Rear Fy mismatch")

    def test_grid_sweep(self):
        slip_angles = np.linspace(-0.3, 0.3, 7)
        slip_ratios = np.linspace(-0.2, 0.4, 7)

        wp_torch = self.tire_torch.front_tire_model_parameters()
        wp_casadi = self._get_tire_params_dict(self.tire_casadi.front_tire_model_parameters)

        sa_sym = ca.MX.sym('sa')
        sr_sym = ca.MX.sym('sr')
        fx_sym, fy_sym = self.tire_casadi.tire_forces_model(sa_sym, sr_sym, wp_casadi)
        f_casadi = ca.Function('f', [sa_sym, sr_sym], [fx_sym, fy_sym])

        for sa_val in slip_angles:
            for sr_val in slip_ratios:
                sa_t = torch.tensor([[sa_val]], dtype=torch.float32)
                sr_t = torch.tensor([[sr_val]], dtype=torch.float32)

                fx_t, fy_t = LinearTireModel.tire_forces_model(sa_t, sr_t, wp_torch)
                res = f_casadi(sa_val, sr_val)

                np.testing.assert_allclose(
                    fx_t.detach().numpy().flatten(),
                    np.array(res[0]).flatten(),
                    rtol=1e-5, atol=1e-7,
                    err_msg=f"Fx mismatch at sa={sa_val:.3f}, sr={sr_val:.3f}"
                )
                np.testing.assert_allclose(
                    fy_t.detach().numpy().flatten(),
                    np.array(res[1]).flatten(),
                    rtol=1e-5, atol=1e-7,
                    err_msg=f"Fy mismatch at sa={sa_val:.3f}, sr={sr_val:.3f}"
                )


if __name__ == '__main__':
    unittest.main()
