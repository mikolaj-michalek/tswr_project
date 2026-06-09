import torch
from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.linear_params import LinearParameters

class LinearTireModel(BaseTireModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = LinearParameters()
        self.rear_tire_model_parameters = LinearParameters()

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        """
        Implements a linear tire model with saturation.
        Returns: normalized Fx, normalized Fy
        """
        
        # 1. Linear Longitudinal Force: Fx = C_x * kappa
        fric_x = wp.C_x * slip_ratio
        
        # 2. Linear Lateral Force: Fy = -C_alpha * alpha
        # The negative sign ensures the force opposes the slip angle
        fric_y = -wp.C_alpha * slip_angle_rad
        
        # 3. Saturation (Friction Circle)
        # We ensure the resultant force does not exceed the friction limit (mu)
        f_total = torch.sqrt(fric_x**2 + fric_y**2) + 1e-8
        
        # Scale forces back if they exceed mu
        scale = torch.where(f_total > wp.mu, wp.mu / f_total, torch.ones_like(f_total))
        
        fric_x = fric_x * scale
        fric_y = fric_y * scale

        return fric_x, fric_y