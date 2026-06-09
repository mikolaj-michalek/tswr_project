import torch

from ldm.systems.car.constants import MIN_VX
from ldm.systems.car.dynamics.state_wrapper import StateWrapper

class VWGolfSingleTrack(torch.nn.Module):
    def __init__(self, vehicle_parameters, tire_model) -> None:
        super(VWGolfSingleTrack, self).__init__()
        # Parameters wrappers
        #self.log_friction = torch.nn.Parameter(torch.tensor([1.0]).log(), requires_grad=True)
        self.log_friction = torch.nn.Parameter(torch.tensor([2.5]).log(), requires_grad=True)
        self.vehicle_parameters = vehicle_parameters
        self.tire_model = tire_model
        self.min_state = torch.tensor([MIN_VX, -torch.inf, -torch.inf])  # v_x, v_y, r

    def forward(self, t, x, u):      
        x = torch.clamp(x, min=self.min_state)
        x_and_u = torch.cat([x, u], dim=-1)

        wx = StateWrapper(x_and_u)

        wp = self.vehicle_parameters()

        tire_forces = self.tire_model(x_and_u, wp) * self.log_friction.exp().unsqueeze(-1)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        
        F_drag = wp.Cd0 * torch.sign(wx.v_x) +\
            wp.Cd1 * wx.v_x +\
            wp.Cd2 * wx.v_x * wx.v_x

        v_x_dot = 1.0 / wp.m * (Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + wp.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / wp.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - wp.m * wx.v_x * wx.r)  #

        r_dot = 1.0 / wp.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * wp.lf - Fy_r * wp.lr)

        return torch.stack([v_x_dot, v_y_dot, r_dot], dim=-1), tire_forces

    @staticmethod
    def state_weights():
        return torch.tensor([
            1.0,  # v_x
            1.0,  # v_y
            1.0,  # r
        ])

    @staticmethod
    def get_state_names():
        return ["x", "y", "yaw", "v_x", "v_y", "r"]
    
    @staticmethod
    def get_control_names():
        return ["omega_wheels_rear", "omega_wheels_front", "delta"]

    def get_parameters(self):
        return {
            'friction': self.log_friction.exp(),
            'vehicle_parameters': self.vehicle_parameters(),
            'tire_parameters': self.tire_model.get_parameters(),
        }

    def get_parameters_vector(self):
        return torch.cat([
            self.log_friction.exp().unsqueeze(0),
            self.vehicle_parameters.get_parameters_vector(),
            self.tire_model.get_parameters_vector(),
        ], dim=0)