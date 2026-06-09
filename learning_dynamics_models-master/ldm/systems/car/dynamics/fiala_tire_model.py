import torch
from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.fiala_params import FialaParameters

class FialaTireModel(BaseTireModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = FialaParameters()
        self.rear_tire_model_parameters = FialaParameters()

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        """
        Implements the Fiala tire model equations.
        Returns: normalized Fx, normalized Fy
        """
        # 1. Longitudinal Force (Simplified Linear Model)
        # Usually Fiala is specific to lateral, so we use a stiffness-based Fx
        fric_x = wp.C_x * slip_ratio
        # Clamp Fx by the friction ellipse limit
        fric_x = torch.clamp(fric_x, min=-wp.mu, max=wp.mu)

        # 2. Lateral Force (Fiala Model)
        alpha = slip_angle_rad
        tan_alpha = torch.tan(alpha)
        
        # Critical slip angle where full sliding begins
        # alpha_c = atan(3 * mu * Fz / C_alpha)
        # Since we return normalized forces (divided by Fz), the formula simplifies:
        z = (wp.C_alpha * torch.abs(tan_alpha)) / (3 * wp.mu + 1e-8)
        
        # Piecewise logic for Elastic vs Sliding regions
        # If z < 1.0: Elastic/Transition
        # If z >= 1.0: Full Sliding
        
        fy_elastic = -wp.mu * (z - (1/3)*(z**2) + (1/27)*(z**3)) * torch.sign(alpha)
        fy_sliding = -wp.mu * torch.sign(alpha)
        
        fric_y = torch.where(z < 1.0, fy_elastic, fy_sliding)

        return fric_x, fric_y