import numpy as np
import casadi as ca
import torch

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.casadi_dynamics.kicajka_utils import compute_force_casadi
from ldm.systems.car.dynamics.kicajka2_tire_model import Kicajka2TireModel
from ldm.systems.car.dynamics.kicajka2_params import Kicajka2Parameters
from ldm.utils.bspline import BSpline
from ldm.utils.casadi_bspline import build_casadi_bspline


class CasadiKicajka2TireModel(CasadiBaseTireModel):
    def __init__(self, n, n_up, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = Kicajka2Parameters(n, n_up)
        self.rear_tire_model_parameters = Kicajka2Parameters(n, n_up)

        N = 64
        self.torch_bspline = BSpline(n=n + 2, d=3, num_T_pts=N, name="kicajka2")
        #self.casadi_bspline_x = build_casadi_bspline(self.torch_bspline.u,
        #                                             self.front_tire_model_parameters().cps_x.detach().numpy(),
        #                                             self.torch_bspline.d)
        #self.casadi_bspline_y = build_casadi_bspline(self.torch_bspline.u,
        #                                             self.front_tire_model_parameters().cps_y.detach().numpy(),
        #                                             self.torch_bspline.d)
        basis_function_values = self.torch_bspline.N[0].T
        basis_function_values_flat = basis_function_values.ravel(order='F')
        self.bspline = ca.interpolant(
            'f_kicajka2',
            'bspline',
            [np.linspace(0.0, 1.0, N)],
            basis_function_values_flat,
        )

    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        Sx_norm = slip_ratio / wp['x_norm']
        Alpha_norm = slip_angle_rad / wp['y_norm']

        S_resultant = ca.sqrt(Sx_norm**2 + Alpha_norm**2) + 1e-8

        Sx_mod = S_resultant * wp['x_norm']
        Alpha_mod = S_resultant * wp['y_norm']

        #Fx = compute_force_casadi(self.casadi_bspline_x, Sx_mod, wp['x_scale'], Sx_norm, S_resultant, wp['cps_x'])
        #Fy = compute_force_casadi(self.casadi_bspline_y, Alpha_mod, wp['y_scale'], Alpha_norm, S_resultant, wp['cps_y'])
        Fx = compute_force_casadi(self.bspline, Sx_mod, wp['x_scale'], Sx_norm, S_resultant, wp['cps_x'])
        Fy = compute_force_casadi(self.bspline, Alpha_mod, wp['y_scale'], Alpha_norm, S_resultant, wp['cps_y'])
        return Fx, -Fy


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    #n = 10
    n = 5
    n_up = 0
    torch_model = Kicajka2TireModel(n=n, n_up=n_up)
    casadi_model = CasadiKicajka2TireModel(n=n, n_up=n_up)

    # Use exactly the same trained parameters for both implementations.
    casadi_model.front_tire_model_parameters = torch_model.front_tire_model_parameters

    wp_torch = torch_model.front_tire_model_parameters()
    wp_casadi = torch_model.front_tire_model_parameters.get_parameters_dict()

    sa_sym = ca.MX.sym('sa')
    sr_sym = ca.MX.sym('sr')
    fx_sym, fy_sym = casadi_model.tire_forces_model(sa_sym, sr_sym, wp_casadi)
    tire_force_func = ca.Function('kicajka2_tire_force', [sa_sym, sr_sym], [fx_sym, fy_sym])

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

    print("CasADi vs Torch Kicajka2 numerical check passed.")
    print(f"max |Fx_torch - Fx_casadi| = {fx_abs_err_max:.3e}")
    print(f"max |Fy_torch - Fy_casadi| = {fy_abs_err_max:.3e}")