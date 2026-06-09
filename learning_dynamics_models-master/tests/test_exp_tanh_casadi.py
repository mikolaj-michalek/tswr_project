import torch
import numpy as np
import casadi as ca
import unittest

from ldm.systems.car.casadi_dynamics.exp_tanh_tire_model import CasadiExpTanhTireModel
from ldm.systems.car.dynamics.exp_tanh_tire_model import ExpTanhTireModel


class TestExpTanhTireModelEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tire_torch = ExpTanhTireModel()
        self.tire_casadi = CasadiExpTanhTireModel()

        # Sync parameters
        self.tire_casadi._torch_model.load_state_dict(
            self.tire_torch.state_dict()
        )

    def _params_dict(self, prefix):
        """Extract exponentiated param dict from the torch model for one axle."""
        return CasadiExpTanhTireModel.torch_params_to_dict(self.tire_torch, prefix)

    def _compare_single_tire(self, Fz_val, sa_val, sr_val, prefix, label=""):
        """Compare compute_single_tire_forces for torch vs casadi."""
        # --- Torch ---
        Fz_t = torch.tensor([[Fz_val]], dtype=torch.float32)
        sa_t = torch.tensor([[sa_val]], dtype=torch.float32)
        sr_t = torch.tensor([[sr_val]], dtype=torch.float32)

        fx_t, fy_t = self.tire_torch.compute_single_tire_forces(
            Fz_t, sa_t, sr_t,
            getattr(self.tire_torch, f'a1_{prefix}'),
            getattr(self.tire_torch, f'a2_{prefix}'),
            getattr(self.tire_torch, f'log_a3_{prefix}'),
            getattr(self.tire_torch, f'log_a4_{prefix}'),
            getattr(self.tire_torch, f'a5_{prefix}'),
            getattr(self.tire_torch, f's1_{prefix}'),
            getattr(self.tire_torch, f's2_{prefix}'),
        )
        fx_t_np = fx_t.detach().numpy().flatten()
        fy_t_np = fy_t.detach().numpy().flatten()

        # --- CasADi ---
        Fz_sym = ca.MX.sym('Fz')
        sa_sym = ca.MX.sym('sa')
        sr_sym = ca.MX.sym('sr')
        p = self._params_dict(prefix)

        fx_sym, fy_sym = CasadiExpTanhTireModel.compute_single_tire_forces(
            Fz_sym, sa_sym, sr_sym,
            p['a1'], p['a2'], p['a3'], p['a4'], p['a5'], p['s1'], p['s2'],
        )
        f_casadi = ca.Function('f', [Fz_sym, sa_sym, sr_sym], [fx_sym, fy_sym])
        res = f_casadi(Fz_val, sa_val, sr_val)

        fx_c_np = np.array(res[0]).flatten()
        fy_c_np = np.array(res[1]).flatten()

        np.testing.assert_allclose(
            fx_t_np, fx_c_np, rtol=1e-5, atol=1e-7,
            err_msg=f"Fx mismatch ({label})"
        )
        np.testing.assert_allclose(
            fy_t_np, fy_c_np, rtol=1e-5, atol=1e-7,
            err_msg=f"Fy mismatch ({label})"
        )

    # ---- Individual operating-point tests ----

    def test_nominal_front(self):
        self._compare_single_tire(500.0, 0.05, 0.03, 'f', "nominal front")

    def test_nominal_rear(self):
        self._compare_single_tire(600.0, 0.08, 0.06, 'r', "nominal rear")

    def test_high_slip_angle(self):
        self._compare_single_tire(500.0, 0.3, 0.01, 'f', "high slip angle")

    def test_high_slip_ratio(self):
        self._compare_single_tire(500.0, 0.01, 0.3, 'f', "high slip ratio")

    def test_negative_slip_angle(self):
        self._compare_single_tire(500.0, -0.1, 0.05, 'f', "negative sa")

    def test_negative_slip_ratio(self):
        self._compare_single_tire(500.0, 0.1, -0.05, 'r', "negative sr")

    def test_both_large(self):
        self._compare_single_tire(500.0, 0.4, 0.4, 'f', "both large")

    def test_small_slips(self):
        self._compare_single_tire(500.0, 0.001, 0.001, 'r', "small slips")

    # ---- Grid sweep ----

    def test_grid_sweep(self):
        """Sweep over many operating points for front tire."""
        slip_angles = np.linspace(-0.3, 0.3, 7)
        slip_ratios = np.linspace(-0.2, 0.4, 7)
        Fz_val = 500.0

        p = self._params_dict('f')
        Fz_sym = ca.MX.sym('Fz')
        sa_sym = ca.MX.sym('sa')
        sr_sym = ca.MX.sym('sr')
        fx_sym, fy_sym = CasadiExpTanhTireModel.compute_single_tire_forces(
            Fz_sym, sa_sym, sr_sym,
            p['a1'], p['a2'], p['a3'], p['a4'], p['a5'], p['s1'], p['s2'],
        )
        f_casadi = ca.Function('f', [Fz_sym, sa_sym, sr_sym], [fx_sym, fy_sym])

        for sa_val in slip_angles:
            for sr_val in slip_ratios:
                Fz_t = torch.tensor([[Fz_val]], dtype=torch.float32)
                sa_t = torch.tensor([[sa_val]], dtype=torch.float32)
                sr_t = torch.tensor([[sr_val]], dtype=torch.float32)

                fx_t, fy_t = self.tire_torch.compute_single_tire_forces(
                    Fz_t, sa_t, sr_t,
                    self.tire_torch.a1_f, self.tire_torch.a2_f,
                    self.tire_torch.log_a3_f, self.tire_torch.log_a4_f,
                    self.tire_torch.a5_f, self.tire_torch.s1_f, self.tire_torch.s2_f,
                )
                res = f_casadi(Fz_val, sa_val, sr_val)

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
