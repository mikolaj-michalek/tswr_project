import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.tmeasy_params import TMeasyParameters


class TMeasyTireModel(BaseTireModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = TMeasyParameters()
        self.rear_tire_model_parameters = TMeasyParameters()

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        # Implementation of TMeasy "User-Appropriate Tyre-Modelling"
        #
        
        # 1. Coordinate Transforms
        # TMeasy uses lateral slip sy roughly equivalent to tan(alpha)
        # and longitudinal slip sx (slip_ratio)
        sx = slip_ratio
        sy = torch.tan(slip_angle_rad) 

        # 2. Normalization Factors (Effective Slip at Peak)
        # We use s_m as the normalization factor so that peak occurs at normalized slip = 1.
        s_nx = sx / (wp.s_mx + 1e-8)
        s_ny = sy / (wp.s_my + 1e-8)

        # 3. Generalized Slip (Normalized Combined Slip)
        # Eq (13) in paper
        s_norm = torch.sqrt(s_nx**2 + s_ny**2) + 1e-8

        # 4. Angle functions for interpolation
        # Eq (15) in paper
        # Note: s_nx / s_norm = cos_phi, s_ny / s_norm = sin_phi
        cos_phi = s_nx / s_norm
        sin_phi = s_ny / s_norm

        # 5. Interpolate Parameters for Combined Curve
        # Eq (14) in paper
        
        # Interpolated Peak Friction (mu_M)
        mu_M = torch.sqrt((wp.mu_mx * cos_phi)**2 + (wp.mu_my * sin_phi)**2)
        
        # Interpolated Sliding Friction (mu_G)
        mu_G = torch.sqrt((wp.mu_gx * cos_phi)**2 + (wp.mu_gy * sin_phi)**2)
        
        # Interpolated Initial Stiffness (df_0)
        # Note: We scale stiffness by the slip normalization factors (s_mx, s_my) 
        # because df_0 in the paper is dF/ds, and we are working in normalized slip s/s_m.
        # The effective stiffness in the normalized domain is df_0 * s_m.
        df0_norm = torch.sqrt((wp.df_x0 * wp.s_mx * cos_phi)**2 + (wp.df_y0 * wp.s_my * sin_phi)**2)
        
        # Interpolated Sliding Slip (s_G_norm)
        # Since we are in normalized space, the peak is at 1.0.
        # We normalize the sliding slip parameters similarly.
        s_gx_norm = wp.s_gx / wp.s_mx
        s_gy_norm = wp.s_gy / wp.s_my
        s_G_norm = torch.sqrt((s_gx_norm * cos_phi)**2 + (s_gy_norm * sin_phi)**2)

        # 6. Calculate Resultant Force F(s) (Normalized as Friction Coefficient)
        # The curve is defined in two regions: Adhesion (0 to s_M) and Sliding (s_M to s_G)
        # In our normalized space, s_M = 1.0.
        
        # --- Region 1: Adhesion (s_norm <= 1.0) ---
        # Rational function: F = (df0 * s) / (1 + b*s + c*s^2)
        # Conditions: F(1) = mu_M, F'(1) = 0
        # Derived parameters: c = 1, b = (df0/mu_M) - 2
        # Check for stability: df0 >= 2*mu_M is required for b >= 0
        
        b = (df0_norm / (mu_M + 1e-8)) - 2.0
        # Clamp b to be non-negative to ensure physical validity (no poles)
        b = torch.clamp(b, min=0.0) 
        
        F_adhesion = (df0_norm * s_norm) / (1.0 + b * s_norm + s_norm**2 + 1e-8)

        # --- Region 2: Sliding (1.0 < s_norm <= s_G_norm) ---
        # Cubic polynomial interpolation between Peak and Sliding
        # Conditions: F(1)=mu_M, F'(1)=0, F(s_G)=mu_G, F'(s_G)=0
        
        # Normalized coordinate within sliding region [0, 1]
        t = (s_norm - 1.0) / (s_G_norm - 1.0 + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        
        # Hermite spline step: H = 3t^2 - 2t^3 (0 to 1) -> drops from 0 to -1? 
        # Standard formulation: F = F_start + (F_end - F_start) * (3t^2 - 2t^3)
        blend = 3.0 * t**2 - 2.0 * t**3
        F_sliding_trans = mu_M + (mu_G - mu_M) * blend
        
        # --- Region 3: Full Sliding (s_norm > s_G_norm) ---
        F_full_sliding = mu_G

        # Combine Regions
        mu_total = torch.where(
            s_norm <= 1.0,
            F_adhesion,
            torch.where(
                s_norm <= s_G_norm,
                F_sliding_trans,
                F_full_sliding
            )
        )

        # 7. Project back to X and Y
        # Fx = F * cos_phi, Fy = F * sin_phi
        mu_x = mu_total * cos_phi
        mu_y = mu_total * sin_phi
        
        return mu_x, -mu_y