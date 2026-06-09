import torch

from ldm.systems.car.dynamics.state_wrapper import StateWrapper


class BaseTireModel(torch.nn.Module):
    def __init__(self):
        super(BaseTireModel, self).__init__()
        self.front_tire_model_parameters = NotImplementedError()
        self.rear_tire_model_parameters = NotImplementedError()

    @staticmethod
    def lf(wp):
        return wp.L - wp.lr

    @staticmethod
    def slip_angle_front_func(wx, wp):        
        return torch.atan((wx.v_y + wp.lf * wx.r) / (wx.v_x + wp.eps)) - wx.delta

    @staticmethod
    def slip_angle_rear_func(wx, wp):
        return torch.atan((wx.v_y - wp.lr * wx.r) / (wx.v_x + wp.eps))

    @staticmethod
    def slip_ratio_front_func(wx, wp):
        v_front = wx.v_x * \
            torch.cos(wx.delta) + (wx.v_y + wx.r *
                                   wp.lf) * torch.sin(wx.delta)
        slip_ratio = (wx.omega_wheels_front - v_front) / \
            (torch.maximum(wx.omega_wheels_front, v_front) + wp.eps)
        return slip_ratio

    @staticmethod
    def slip_ratio_rear_func(wx, wp):
        v_rear = wx.v_x
        slip_ratio = (wx.omega_wheels_rear - v_rear) / \
            (torch.maximum(wx.omega_wheels_rear, v_rear) + wp.eps)
        return slip_ratio

    @staticmethod
    def Fz_front(wp):
        return wp.m * wp.g * wp.lr / wp.L   
    
    @staticmethod
    def Fz_rear(wp):
        return wp.m * wp.g * wp.lf / wp.L

    def tire_forces_model(self, slip_angle, slip_ratio, wp):
        raise NotImplementedError()

    def tire_forces(self, slip_ratio_front, slip_angle_front,
                      slip_ratio_rear, slip_angle_rear, wp_st):
        fric_xf, fric_yf = self.tire_forces_model(slip_angle_front, slip_ratio_front,
                                                  self.front_tire_model_parameters())
        fric_xr, fric_yr = self.tire_forces_model(slip_angle_rear, slip_ratio_rear,
                                                  self.rear_tire_model_parameters())
        Fxr = self.Fz_rear(wp_st) * fric_xr
        Fyr = self.Fz_rear(wp_st) * fric_yr
        Fxf = self.Fz_front(wp_st) * fric_xf
        Fyf = self.Fz_front(wp_st) * fric_yf
        return Fyf, Fyr, Fxf, Fxr

    def forward(self, x, wp_st):
        wx = StateWrapper(x)
        slip_angle_front = self.slip_angle_front_func(wx, wp_st)
        slip_ratio_front = self.slip_ratio_front_func(wx, wp_st)
        slip_angle_rear = self.slip_angle_rear_func(wx, wp_st)
        slip_ratio_rear = self.slip_ratio_rear_func(wx, wp_st)
        tire_forces = self.tire_forces(slip_ratio_front, slip_angle_front,
                                       slip_ratio_rear, slip_angle_rear, wp_st)
        slips = torch.stack([slip_angle_front, slip_ratio_front,
                             slip_angle_rear, slip_ratio_rear], dim=-1)
        return torch.stack(tire_forces, dim=-1), slips

    def get_parameters(self):
        return {
            'front_tire': self.front_tire_model_parameters(),
            'rear_tire': self.rear_tire_model_parameters(),
        }

    def get_parameters_vector(self):
        return torch.cat([
            self.front_tire_model_parameters.get_parameters_vector(),
            self.rear_tire_model_parameters.get_parameters_vector()
        ], dim=0)
        
        