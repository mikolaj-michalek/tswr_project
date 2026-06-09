import numpy as np
import casadi as ca

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.dynamics.pacejka_params import PacejkaParameters


class CasadiPacejkaTireModel(CasadiBaseTireModel):
    def __init__(self):
        super().__init__()
        self.front_tire_model_parameters = PacejkaParameters()
        self.rear_tire_model_parameters = PacejkaParameters()


    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        # wp is a dictionary containing Pacejka params (Sx_p, Alpha_p, etc.)
        
        # alpha from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / np.pi
        
        # Normalize
        Sx_norm = slip_ratio / wp['Sx_p']
        Alpha_norm = slip_angle_deg / wp['Alpha_p']
        
        # Resultant slip
        S_resultant = ca.sqrt(Sx_norm**2 + Alpha_norm**2) + 1e-8
        
        # Modified slip factors
        Sx_mod = S_resultant * wp['Sx_p']
        Alpha_mod = S_resultant * wp['Alpha_p']
        
        # Lateral Force (Fy)
        Alpha_final = Alpha_mod
        # The Pacejka Magic Formula
        # y = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
        
        B_y = wp['By']
        C_y = wp['Cy']
        D_y = wp['Dy']
        E_y = wp['Ey'] # Passed as Ey (already converted from log_Ey_plus1 in wrapper if needed)
        
        term_y = B_y * Alpha_final
        val_y = D_y * ca.sin(C_y * ca.atan(term_y - E_y * (term_y - ca.atan(term_y))))
        
        Fy_raw = (Alpha_norm / S_resultant) * val_y
        
        # Longitudinal Force (Fx)
        Sx_final = Sx_mod
        
        B_x = wp['Bx']
        C_x = wp['Cx']
        D_x = wp['Dx']
        E_x = wp['Ex']
        
        term_x = B_x * Sx_final
        val_x = D_x * ca.sin(C_x * ca.atan(term_x - E_x * (term_x - ca.atan(term_x))))
        
        Fx_raw = (Sx_norm / S_resultant) * val_x
        
        # Return Fx, -Fy (matching PyTorch sign convention)
        return Fx_raw, -Fy_raw
