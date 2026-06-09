import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.pacejka_offset_params import PacejkaOffsetParameters

class PacejkaOffsetTireModel(BaseTireModel):
    def __init__(self, randomize_init=0.0, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = PacejkaOffsetParameters(randomize_init=randomize_init)
        self.rear_tire_model_parameters = PacejkaOffsetParameters(randomize_init=randomize_init)

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        #[1] Bakker, E., Nyborg, L., and Pacejka, H.B.
        #   Tyre modelling for use in vehicle dynamics studies. United States: N. p., 1987. Web
        
        #[2] W. F. Milliken and D. L. Milliken,
        #   Race Car Vehicle Dynamics.Warrendale, PA, USA: SAE, 1995.
        
        # https://skill-lync.com/student-projects/Combined-slip-correction-using-Pacejka-tire-model-25918
        # https://www.researchgate.net/publication/344073372_Tire_Modeling_Using_Pacejka_Model
        # Unpack the state variables
        
        # alpha from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / torch.pi
        
        # Calculate normalized slip ratio and slip angle
        #Sx_norm = slip_ratio / wp.Sx_p
        #Alpha_norm = slip_angle_deg / wp.Alpha_p
        Sx_norm = (slip_ratio + wp.Shx) / wp.Sx_p
        Alpha_norm = (slip_angle_deg + wp.Shy) / wp.Alpha_p
        
        # Compute the resultant slip
        S_resultant = torch.sqrt(Sx_norm**2 + Alpha_norm**2) + 1e-8
        
        # Find the modified slip factors
        Sx_mod = S_resultant * wp.Sx_p
        Alpha_mod = S_resultant * wp.Alpha_p
        
        # Calculate the Lateral Force using Pacejka formula
        Alpha_final = Alpha_mod# + wp.Shy
        Fy = ((Alpha_norm / S_resultant) * wp.Dy * torch.sin(wp.Cy * torch.atan((wp.By * Alpha_final) - 
            wp.Ey * (wp.By * Alpha_final - torch.atan(wp.By * Alpha_final))))) + wp.Svy
        
        # Calculate the Longitudinal Force using Pacejka formula
        Sx_final = Sx_mod# + wp.Shx
        Fx = ((Sx_norm / S_resultant) * wp.Dx * torch.sin(wp.Cx * torch.atan((wp.Bx * Sx_final) - 
            wp.Ex * (wp.Bx * Sx_final - torch.atan(wp.Bx * Sx_final))))) + wp.Svx
        
        return Fx, - Fy