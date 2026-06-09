import torch

from ldm.systems.car.dynamics.state_wrapper import ExtendedStateWrapper

class SingleTrackLoadTransfer(torch.nn.Module):

    def __init__(self, vehicle_parameters, tire_model) -> None:
        super(SingleTrackLoadTransfer, self).__init__()
        # Parameters wrappers
        self.vehicle_parameters = vehicle_parameters
        self.tire_model = tire_model
        self.min_state = torch.tensor([0.5, -torch.inf, -torch.inf, -torch.inf])  # v_x, v_y, r
        
    def forward(self, t, x, u):      
        x = torch.clamp(x, min=self.min_state)
        x_and_u = torch.cat([x, u], dim=-1)

        wx = ExtendedStateWrapper(x_and_u)

        assert torch.all(torch.sqrt(wx.v_x**2 + wx.v_y**2) >= 0.05), "Car is not moving"

        wp = self.vehicle_parameters()

        tire_forces = self.tire_model(x_and_u, wp)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        
        F_drag = wp.Cd0 * torch.sign(wx.v_x) +\
            wp.Cd1 * wx.v_x +\
            wp.Cd2 * wx.v_x * wx.v_x

        Fx_net = Fx_r + Fx_f * torch.cos(wx.delta) - Fy_f * torch.sin(wx.delta)
        Fx_all = Fx_net - F_drag + wp.m * wx.v_y * wx.r
        v_x_dot_ = 1.0 / wp.m * Fx_all
        v_x_dot = wp.vxtau * (wp.vxk * v_x_dot_ - wx.v_x)

        v_y_dot_ = 1.0 / wp.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - wp.m * wx.v_x * wx.r)  #
        v_y_dot = wp.vytau * (wp.vyk * v_y_dot_ - wx.v_y)

        r_dot_ = 1.0 / wp.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * wp.lf - Fy_r * wp.lr)
        r_dot = wp.rtau * (wp.rk * r_dot_ - wx.r)

        # v1
        #dFz_dot = - wp.K_lt * (wx.dFz - wp.CoM_height / (wp.L) * Fx_net)
        # v2
        #dFz_dot = - wp.K_lt * wx.dFz + wp.CoM_height / (wp.L) * Fx_net
        dFz_dot = - wp.K_lt * wx.dFz + wp.CoM_height / (wp.L) * Fx_all

        # omega_wheels_dot = wp.R / wp.I_e * (wp.K_fi * wx.Iq - wp.R * Fx_f - wp.R * Fx_r
        #                                   - wx.omega_wheels * wp.b1 - torch.sign(wx.omega_wheels) * wp.b0)

        return torch.stack([v_x_dot, v_y_dot, r_dot, dFz_dot], dim=-1), tire_forces
 

    @staticmethod
    def state_weights():
        return torch.tensor([
            1.0,  # v_x
            1.0,  # v_y
            1.0,  # r
            0.0,  # dFz
        ])
        
    @staticmethod
    def get_state_names():
        return ["v_x", "v_y", "r", "dFz"]

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