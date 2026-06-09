import numpy as np
import casadi as ca

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.casadi_dynamics.kicajka_utils import compute_force_casadi
from ldm.systems.car.dynamics.kicajka4_params import Kicajka4Parameters
from ldm.utils.bspline import BSpline

class CasadiKicajka4TireModel(CasadiBaseTireModel):
    def __init__(self, n, n_up, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = Kicajka4Parameters(n, n_up)
        self.rear_tire_model_parameters = Kicajka4Parameters(n, n_up)

        N = 64
        self.torch_bspline = BSpline(n=n + 2, d=3, num_T_pts=N, name="kicajka4")
        basis_function_values = self.torch_bspline.N[0].T
        basis_function_values_flat = basis_function_values.ravel(order='F')
        self.bspline = ca.interpolant(
            'f_kicajka4',
            'bspline',
            [np.linspace(0.0, 1.0, N)],
            basis_function_values_flat,
        )

    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        Sx_norm = (slip_ratio + wp['sr_offset']) / wp['x_norm']
        Alpha_norm = (slip_angle_rad + wp['sa_offset']) / wp['y_norm']
        
        # Compute the resultant slip
        S_resultant = ca.sqrt(Sx_norm**2 + Alpha_norm**2) + 1e-8
        
        # Find the modified slip factors
        Sx_mod = S_resultant * wp['x_norm']
        Alpha_mod = S_resultant * wp['y_norm']

        Fx = compute_force_casadi(self.bspline, Sx_mod, wp['x_scale'], Sx_norm, S_resultant, wp['cps_x'])
        Fy = compute_force_casadi(self.bspline, Alpha_mod, wp['y_scale'], Alpha_norm, S_resultant, wp['cps_y'])
        return Fx, - Fy