import torch

from ldm.systems.car.constants import MIN_VX
from ldm.systems.car.dynamics.state_wrapper import StateWrapper

class SingleTrack(torch.nn.Module):
    def __init__(self, vehicle_parameters, tire_model) -> None:
        super(SingleTrack, self).__init__()
        # Parameters wrappers
        self.vehicle_parameters = vehicle_parameters
        self.tire_model = tire_model
        self.register_buffer('min_state', torch.tensor([MIN_VX, -torch.inf, -torch.inf, -torch.inf, -torch.inf]))  # v_x, v_y, r, friction_front, friction_rear

    def forward(self, t, x, u):      
        x = torch.clamp(x, min=self.min_state)
        x_and_u = torch.cat([x, u], dim=-1)

        wx = StateWrapper(x_and_u)
        wp = self.vehicle_parameters()

        normalized_tire_forcses, slips = self.tire_model(x_and_u, wp)

        Fy_f_n, Fy_r_n, Fx_f_n, Fx_r_n = torch.unbind(normalized_tire_forcses, dim=-1)
        Fy_f = Fy_f_n * wx.friction_front
        Fy_r = Fy_r_n * wx.friction_rear
        Fx_f = Fx_f_n * wx.friction_front
        Fx_r = Fx_r_n * wx.friction_rear
        tire_forces = torch.stack([Fy_f, Fy_r, Fx_f, Fx_r], dim=-1)
        
        F_drag = wp.Cd0 * torch.sign(wx.v_x) + \
            wp.Cd1 * wx.v_x + \
            wp.Cd2 * wx.v_x * wx.v_x

        v_x_dot = 1.0 / wp.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + wp.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / wp.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - wp.m * wx.v_x * wx.r)  #

        r_dot = 1.0 / wp.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * wp.lf - Fy_r * wp.lr)

        friction_front_dot = torch.zeros_like(wx.friction_front)
        friction_rear_dot = torch.zeros_like(wx.friction_rear)

        return torch.stack([v_x_dot, v_y_dot, r_dot, friction_front_dot, friction_rear_dot], dim=-1), tire_forces, slips

    @staticmethod
    def state_weights():
        return torch.tensor([
            1.0,  # v_x
            1.0,  # v_y
            1.0,  # r
            0.0,  # friction_front
            0.0,  # friction_rear
        ])

    @staticmethod
    def get_state_names():
        return ["x", "y", "yaw", "v_x", "v_y", "r", "friction_front", "friction_rear"]
    
    @staticmethod
    def get_control_names():
        return ["omega_wheels_rear", "omega_wheels_front", "delta"]

    def get_parameters(self):
        return {
            'vehicle_parameters': self.vehicle_parameters(),
            'tire_parameters': self.tire_model.get_parameters(),
        }

    def get_parameters_vector(self):
        return torch.cat([
            self.vehicle_parameters.get_parameters_vector(),
            self.tire_model.get_parameters_vector(),
        ], dim=0)

    def compute_slips(self, x, u):
        x = torch.clamp(x, min=self.min_state)
        wx = StateWrapper(torch.cat([x, u], dim=-1))
        wp = self.vehicle_parameters()
        slip_angle_front = self.tire_model.slip_angle_front_func(wx, wp)
        slip_ratio_front = self.tire_model.slip_ratio_front_func(wx, wp)
        slip_angle_rear = self.tire_model.slip_angle_rear_func(wx, wp)
        slip_ratio_rear = self.tire_model.slip_ratio_rear_func(wx, wp)
        return slip_angle_front, slip_ratio_front, slip_angle_rear, slip_ratio_rear