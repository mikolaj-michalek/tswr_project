import torch
from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.dugoff_params import DugoffParameters

class DugoffTireModel(BaseTireModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = DugoffParameters()
        self.rear_tire_model_parameters = DugoffParameters()

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        """
        Implements the Dugoff tire model equations for combined slip.
        Returns: normalized Fx, normalized Fy
        """
        # 1. Calculate the theoretical linear forces
        # Fx_linear = Cx * (kappa / (1 + kappa))
        # Fy_linear = Calphas * (tan(alpha) / (1 + kappa))
        
        # Note: In many implementations, the (1 + kappa) denominator is used
        # to account for the reduction in contact area during high slip.
        #denom = 1.0 + slip_ratio
        denom = 1.0 - slip_ratio
        fx_lin = wp.C_x * (slip_ratio / denom)
        fy_lin = -wp.C_alpha * (torch.tan(slip_angle_rad) / denom)

        # 2. Calculate the "limit" factor lambda (L)
        # L = (mu * Fz * (1 + kappa)) / (2 * sqrt((Cx*kappa)^2 + (Calpha*tan(alpha))^2))
        # Since we use normalized forces (divided by Fz), Fz cancels out:
        resultant_lin = torch.sqrt((wp.C_x * slip_ratio)**2 + 
                                   (wp.C_alpha * torch.tan(slip_angle_rad))**2) + 1e-8
        
        L = (wp.mu * denom) / (2.0 * resultant_lin)

        # 3. Calculate the gain factor f(L)
        # if L < 1: f(L) = L * (2 - L)
        # else: f(L) = 1
        f_L = torch.where(L < 1.0, L * (2.0 - L), torch.ones_like(L))

        # 4. Final Forces
        fric_x = fx_lin * f_L
        fric_y = fy_lin * f_L

        return fric_x, fric_y