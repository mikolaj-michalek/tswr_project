import casadi as ca

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.dynamics.dugoff_params import DugoffParameters


class CasadiDugoffTireModel(CasadiBaseTireModel):
    def __init__(self):
        super().__init__()
        self.front_tire_model_parameters = DugoffParameters()
        self.rear_tire_model_parameters = DugoffParameters()

    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        """
        Implements the Dugoff tire model equations for combined slip (CasADi version).
        wp is a dictionary containing Dugoff params (C_alpha, C_x, mu).
        Returns: Fx, Fy
        """
        # 1. Calculate the theoretical linear forces
        denom = 1.0 - slip_ratio
        fx_lin = wp['C_x'] * (slip_ratio / denom)
        fy_lin = -wp['C_alpha'] * (ca.tan(slip_angle_rad) / denom)

        # 2. Calculate the "limit" factor lambda (L)
        resultant_lin = ca.sqrt((wp['C_x'] * slip_ratio)**2 +
                                (wp['C_alpha'] * ca.tan(slip_angle_rad))**2) + 1e-8

        L = (wp['mu'] * denom) / (2.0 * resultant_lin)

        # 3. Calculate the gain factor f(L)
        # if L < 1: f(L) = L * (2 - L)
        # else: f(L) = 1
        f_L = ca.if_else(L < 1.0, L * (2.0 - L), 1.0)

        # 4. Final Forces
        fric_x = fx_lin * f_L
        fric_y = fy_lin * f_L

        return fric_x, fric_y
