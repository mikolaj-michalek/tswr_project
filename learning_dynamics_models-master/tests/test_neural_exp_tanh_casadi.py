import torch
import numpy as np
import casadi as ca
import unittest

from ldm.systems.car.casadi_dynamics.neural_exp_tanh_tire_model import (
    CasadiNeuralExpTanhTireModel,
    _torch_sequential_to_casadi,
)
from ldm.systems.car.dynamics.neural_exp_tanh_tire_model import NeuralExpTanhTireModel


class TestNeuralExpTanhEquivalence(unittest.TestCase):
    """Verify CasADi neural-ExpTanh tire model matches the PyTorch version."""

    def setUp(self):
        torch.manual_seed(42)
        self.torch_model = NeuralExpTanhTireModel(nn=16)
        self.torch_model.eval()
        self.casadi_model = CasadiNeuralExpTanhTireModel(self.torch_model)

    # ------------------------------------------------------------------
    # Helper: compare compute_single_tire_forces for one axle
    # ------------------------------------------------------------------
    def _compare_single_tire(self, r_val, vx_val, vy_val, Fz_val,
                             sa_val, sr_val, axle='front', label=""):
        # --- Torch ---
        def _t(v):
            return torch.tensor([[v]], dtype=torch.float32)

        if axle == 'front':
            exptanh_fn_torch = self.torch_model.exptanh_params_front
            s1s2_fn_torch = self.torch_model.s1s2_front
            exptanh_fn_ca = self.casadi_model._exptanh_params_front_fn
            s1s2_fn_ca = self.casadi_model._s1s2_front_fn
        else:
            exptanh_fn_torch = self.torch_model.exptanh_params_rear
            s1s2_fn_torch = self.torch_model.s1s2_rear
            exptanh_fn_ca = self.casadi_model._exptanh_params_rear_fn
            s1s2_fn_ca = self.casadi_model._s1s2_rear_fn

        with torch.no_grad():
            fx_t, fy_t = self.torch_model.compute_single_tire_forces(
                _t(r_val), _t(vx_val), _t(vy_val), _t(Fz_val),
                _t(sa_val), _t(sr_val),
                exptanh_fn_torch, s1s2_fn_torch,
            )
        fx_t_np = fx_t.numpy().flatten()
        fy_t_np = fy_t.numpy().flatten()

        # --- CasADi ---
        r_s   = ca.MX.sym('r')
        vx_s  = ca.MX.sym('vx')
        vy_s  = ca.MX.sym('vy')
        Fz_s  = ca.MX.sym('Fz')
        sa_s  = ca.MX.sym('sa')
        sr_s  = ca.MX.sym('sr')

        fx_sym, fy_sym = self.casadi_model.compute_single_tire_forces(
            r_s, vx_s, vy_s, Fz_s, sa_s, sr_s,
            exptanh_fn_ca, s1s2_fn_ca,
        )
        f_ca = ca.Function('f', [r_s, vx_s, vy_s, Fz_s, sa_s, sr_s],
                           [fx_sym, fy_sym])
        res = f_ca(r_val, vx_val, vy_val, Fz_val, sa_val, sr_val)
        fx_c_np = np.array(res[0]).flatten()
        fy_c_np = np.array(res[1]).flatten()

        np.testing.assert_allclose(
            fx_t_np, fx_c_np, rtol=1e-5, atol=1e-6,
            err_msg=f"Fx mismatch ({label})"
        )
        np.testing.assert_allclose(
            fy_t_np, fy_c_np, rtol=1e-5, atol=1e-6,
            err_msg=f"Fy mismatch ({label})"
        )

    # ------------------------------------------------------------------
    # Sub-network sanity checks
    # ------------------------------------------------------------------
    def test_exptanh_params_front_network(self):
        """MLP forward pass: exptanh_params_front."""
        x_in = np.array([0.2, 10.0, 0.05, 500.0])
        x_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_t = self.torch_model.exptanh_params_front(x_t).numpy().flatten()
        y_ca = np.array(
            self.casadi_model._exptanh_params_front_fn(x_in[:, None])
        ).flatten()
        np.testing.assert_allclose(y_t, y_ca, rtol=1e-5, atol=1e-6)

    def test_s1s2_front_network(self):
        """MLP forward pass: s1s2_front."""
        x_in = np.array([0.05, 0.03])
        x_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_t = self.torch_model.s1s2_front(x_t).numpy().flatten()
        y_ca = np.array(
            self.casadi_model._s1s2_front_fn(x_in[:, None])
        ).flatten()
        np.testing.assert_allclose(y_t, y_ca, rtol=1e-5, atol=1e-6)

    def test_exptanh_params_rear_network(self):
        x_in = np.array([0.1, 8.0, -0.02, 600.0])
        x_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_t = self.torch_model.exptanh_params_rear(x_t).numpy().flatten()
        y_ca = np.array(
            self.casadi_model._exptanh_params_rear_fn(x_in[:, None])
        ).flatten()
        np.testing.assert_allclose(y_t, y_ca, rtol=1e-5, atol=1e-6)

    def test_s1s2_rear_network(self):
        x_in = np.array([-0.1, 0.06])
        x_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_t = self.torch_model.s1s2_rear(x_t).numpy().flatten()
        y_ca = np.array(
            self.casadi_model._s1s2_rear_fn(x_in[:, None])
        ).flatten()
        np.testing.assert_allclose(y_t, y_ca, rtol=1e-5, atol=1e-6)

    # ------------------------------------------------------------------
    # Single-tire force tests
    # ------------------------------------------------------------------
    def test_nominal_front(self):
        self._compare_single_tire(0.2, 10.0, 0.5, 500.0, 0.05, 0.03,
                                  'front', 'nominal front')

    def test_nominal_rear(self):
        self._compare_single_tire(0.1, 8.0, -0.3, 600.0, 0.08, 0.06,
                                  'rear', 'nominal rear')

    def test_high_slip_angle(self):
        self._compare_single_tire(0.0, 12.0, 1.0, 500.0, 0.3, 0.01,
                                  'front', 'high sa')

    def test_high_slip_ratio(self):
        self._compare_single_tire(0.0, 12.0, 0.0, 500.0, 0.01, 0.3,
                                  'front', 'high sr')

    def test_negative_slips(self):
        self._compare_single_tire(0.0, 10.0, 0.0, 500.0, -0.1, -0.05,
                                  'rear', 'neg slips')

    def test_large_slips(self):
        self._compare_single_tire(0.3, 15.0, 2.0, 700.0, 0.4, 0.4,
                                  'front', 'large slips')

    # ------------------------------------------------------------------
    # Grid sweep
    # ------------------------------------------------------------------

    
    def test_grid_sweeps(self):
        """Sweep over operating points for the front axle."""
        def grid_sweep(side="front"):
            """Sweep over operating points for the rear axle."""
            N = 200
            r = np.random.uniform(-1.0, 1.0, N)
            vx = np.random.uniform(0.5, 8.0, N)
            vy = np.random.uniform(-2.0, 2.0, N)
            slip_ratios = np.random.uniform(-1.0, 1.0, N)
            slip_angles = np.arctan2(vy, vx)  # Convert to slip angle
            Fz = np.random.uniform(100.0, 700.0, N)

            for i in range(N):
                r_val = r[i]
                vx_val = vx[i]
                vy_val = vy[i]
                sa_val = slip_angles[i]
                sr_val = slip_ratios[i]
                Fz_val = Fz[i]
                self._compare_single_tire(
                        r_val, vx_val, vy_val, Fz_val, sa_val, sr_val,
                        side, f'sa={sa_val:.2f} sr={sr_val:.2f}')
        grid_sweep("front")
        grid_sweep("rear")


if __name__ == '__main__':
    unittest.main()
