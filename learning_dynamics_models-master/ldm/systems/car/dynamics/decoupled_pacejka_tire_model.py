import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.decoupled_pacejka_params import DecoupledPacejkaParameters


class DecoupledPacejkaTireModel(BaseTireModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = DecoupledPacejkaParameters()
        self.rear_tire_model_parameters = DecoupledPacejkaParameters()

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        # Calculate the Lateral Force using Pacejka formula
        Fy = wp.Dy * torch.sin(wp.Cy * torch.atan((wp.By * slip_angle_rad) - 
            wp.Ey * (wp.By * slip_angle_rad - torch.atan(wp.By * slip_angle_rad)))) 
        
        # Calculate the Longitudinal Force using Pacejka formula
        Fx = wp.Dx * torch.sin(wp.Cx * torch.atan((wp.Bx * slip_ratio) - 
            wp.Ex * (wp.Bx * slip_ratio - torch.atan(wp.Bx * slip_ratio)))) 
        
        return Fx, - Fy