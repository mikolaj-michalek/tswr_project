import casadi as ca

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.dynamics.linear_params import LinearParameters


class CasadiLinearTireModel(CasadiBaseTireModel):
    def __init__(self):
        super().__init__()
        self.front_tire_model_parameters = LinearParameters()
        self.rear_tire_model_parameters = LinearParameters()

    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        """
        Implements a linear tire model with saturation (CasADi version).
        wp is a dictionary containing Linear params (C_alpha, C_x, mu).
        Returns: Fx, Fy (normalised by Fz)
        """
        # 1. Linear Longitudinal Force
        fric_x = wp['C_x'] * slip_ratio

        # 2. Linear Lateral Force
        fric_y = -wp['C_alpha'] * slip_angle_rad

        # 3. Saturation (Friction Circle)
        f_total = ca.sqrt(fric_x**2 + fric_y**2) + 1e-8
        scale = ca.if_else(f_total > wp['mu'], wp['mu'] / f_total, 1.0)

        fric_x = fric_x * scale
        fric_y = fric_y * scale

        return fric_x, fric_y
