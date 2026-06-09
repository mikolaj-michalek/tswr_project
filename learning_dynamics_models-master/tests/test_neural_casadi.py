import torch
import numpy as np
import casadi as ca
import unittest

from ldm.systems.car.casadi_dynamics.neural_tire_model import CasadiNeuralTireModel
from ldm.systems.car.dynamics.neural_tire_model import NeuralTireModel
from ldm.systems.car.dynamics.single_track_params import DefaultSingleTrackParameters


class TestNeuralTireModelEquivalence(unittest.TestCase):
    """Verify CasADi NeuralTireModel matches the PyTorch version (vinvariant mode)."""

    def setUp(self):
        torch.manual_seed(42)
        self.torch_model = NeuralTireModel(nn=32, input_type="vinvariant")
        self.torch_model.eval()
        self.casadi_model = CasadiNeuralTireModel(self.torch_model)
        self.vehicle_params_torch = DefaultSingleTrackParameters()

    def _vehicle_params_dict(self):
        """Extract vehicle parameters as a plain dict for CasADi."""
        vp = self.vehicle_params_torch()
        return {k: getattr(vp, k).detach().item() for k in vp._fields if getattr(vp, k) is not None}

    # ------------------------------------------------------------------
    # MLP sanity check
    # ------------------------------------------------------------------
    def test_mlp_forward(self):
        """Standalone MLP forward pass."""
        x_in = np.array([0.05, 0.03, -0.02, 0.01])
        x_t = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_t = self.torch_model.mlp(x_t).numpy().flatten()
        y_ca = np.array(self.casadi_model._mlp_fn(x_in[:, None])).flatten()
        np.testing.assert_allclose(y_t, y_ca, rtol=1e-5, atol=1e-6)

    # ------------------------------------------------------------------
    # Full forward comparison
    # ------------------------------------------------------------------
    def _compare_forward(self, vx, vy, r, omega_r, omega_f, delta, label=""):
        """Run both models on the same state+control and compare forces."""
        x_u_np = np.array([vx, vy, r, omega_r, omega_f, delta], dtype=np.float64)
        x_u_torch = torch.tensor(x_u_np, dtype=torch.float32).unsqueeze(0)

        # Torch
        wp_torch = self.vehicle_params_torch()
        with torch.no_grad():
            forces_t = self.torch_model.forward(x_u_torch, wp_torch)
        forces_t_np = forces_t.numpy().flatten()

        # CasADi
        wp_ca = self._vehicle_params_dict()
        x_sym = ca.MX.sym('xu', 6)
        forces_sym = self.casadi_model.forward(x_sym, wp_ca)
        f_ca = ca.Function('f', [x_sym], [forces_sym])
        forces_ca_np = np.array(f_ca(x_u_np)).flatten()

        np.testing.assert_allclose(
            forces_t_np, forces_ca_np, rtol=1e-5, atol=1e-6,
            err_msg=f"Forces mismatch ({label})"
        )

    def test_nominal(self):
        self._compare_forward(10.0, 0.5, 0.2, 12.0, 12.0, 0.05, "nominal")

    def test_straight_line(self):
        self._compare_forward(8.0, 0.0, 0.0, 10.0, 10.0, 0.0, "straight")

    def test_high_yaw_rate(self):
        self._compare_forward(10.0, 1.0, 0.8, 12.0, 12.0, 0.1, "high yaw")

    def test_braking(self):
        self._compare_forward(10.0, 0.0, 0.0, 5.0, 5.0, 0.0, "braking")

    def test_negative_steering(self):
        self._compare_forward(10.0, -0.3, -0.1, 12.0, 12.0, -0.1, "neg steer")

    def test_low_speed(self):
        self._compare_forward(3.0, 0.1, 0.05, 4.0, 4.0, 0.02, "low speed")

    # ------------------------------------------------------------------
    # Grid sweep over steering + yaw rate
    # ------------------------------------------------------------------
    def test_grid_sweep(self):
        deltas = np.linspace(-0.15, 0.15, 5)
        yaw_rates = np.linspace(-0.5, 0.5, 5)
        for d in deltas:
            for r_val in yaw_rates:
                self._compare_forward(
                    10.0, 0.3, r_val, 12.0, 12.0, d,
                    f"delta={d:.2f} r={r_val:.2f}")


class TestNeuralTireModelStateMode(unittest.TestCase):
    """Verify CasADi NeuralTireModel in 'state' input mode."""

    def setUp(self):
        torch.manual_seed(99)
        self.torch_model = NeuralTireModel(n_in=6, nn=32, input_type="state")
        self.torch_model.eval()
        self.casadi_model = CasadiNeuralTireModel(self.torch_model)
        self.vehicle_params_torch = DefaultSingleTrackParameters()

    def _vehicle_params_dict(self):
        vp = self.vehicle_params_torch()
        return {k: getattr(vp, k).detach().item() for k in vp._fields if getattr(vp, k) is not None}

    def _compare_forward(self, vx, vy, r, omega_r, omega_f, delta, label=""):
        x_u_np = np.array([vx, vy, r, omega_r, omega_f, delta], dtype=np.float64)
        x_u_torch = torch.tensor(x_u_np, dtype=torch.float32).unsqueeze(0)

        wp_torch = self.vehicle_params_torch()
        with torch.no_grad():
            forces_t = self.torch_model.forward(x_u_torch, wp_torch)
        forces_t_np = forces_t.numpy().flatten()

        wp_ca = self._vehicle_params_dict()
        x_sym = ca.MX.sym('xu', 6)
        forces_sym = self.casadi_model.forward(x_sym, wp_ca)
        f_ca = ca.Function('f', [x_sym], [forces_sym])
        forces_ca_np = np.array(f_ca(x_u_np)).flatten()

        np.testing.assert_allclose(
            forces_t_np, forces_ca_np, rtol=1e-5, atol=1e-6,
            err_msg=f"Forces mismatch ({label})"
        )

    def test_nominal_state(self):
        self._compare_forward(10.0, 0.5, 0.2, 12.0, 12.0, 0.05, "state nominal")

    def test_grid_sweep_state(self):
        deltas = np.linspace(-0.15, 0.15, 5)
        yaw_rates = np.linspace(-0.5, 0.5, 5)
        for d in deltas:
            for r_val in yaw_rates:
                self._compare_forward(
                    10.0, 0.3, r_val, 12.0, 12.0, d,
                    f"delta={d:.2f} r={r_val:.2f}")


if __name__ == '__main__':
    unittest.main()
