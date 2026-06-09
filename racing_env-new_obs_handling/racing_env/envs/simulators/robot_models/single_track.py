import torch
from racing_env.utils.state_wrapper import StateWrapper, ParamWrapper, TireWrapper
from racing_env.envs.simulators.robot_models.pacejka_params import PacejkaParameters
from racing_env.envs.simulators.robot_models.pacejka_tire_model import PacejkaTireModel
from racing_env.envs.simulators.robot_models.base_tire_model import BaseTireModel
from racing_env.envs.simulators.robot_models.single_track_params import VehicleParameters

class SingleTrackPacejkaModel(torch.nn.Module):

    def __init__(self) -> None:
        super(SingleTrackPacejkaModel, self).__init__()
        self.eps = 1e-6
        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()

        self.tire_model = PacejkaTireModel()

    def forward(self, t, x, p_vehicle, p_tire_front, p_tire_rear):
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = self.vehicle_parameters(p_vehicle)
        wx = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tire_front)
        wp_tire_r = self.tire_model_parameters(p_tire_rear)

        # Slip angles and ratios (stored during integration)
        alpha_f = BaseTireModel.slip_angle_front_func(wx, p)
        alpha_r = BaseTireModel.slip_angle_rear_func(wx, p)
        slip_ratio_f = BaseTireModel.slip_ratio_front_func(wx, p)
        slip_ratio_r = BaseTireModel.slip_ratio_func(wx, p)

        # Tire Forces (per-axle friction)
        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r)
        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        Fy_f, Fx_f = Fy_f * wx.front_friction, Fx_f * wx.front_friction
        Fy_r, Fx_r = Fy_r * wx.rear_friction, Fx_r * wx.rear_friction

        self.last_tire_dynamics = {
            "Fx_f": Fx_f, "Fx_r": Fx_r, "Fy_f": Fy_f, "Fy_r": Fy_r,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "slip_ratio_f": slip_ratio_f, "slip_ratio_r": slip_ratio_r,
        }

        # Vehicle Dynamics

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - p.m * wx.v_x * wx.r)

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * p.lf - Fy_r * p.lr)

        omega_wheels_dot = (wx.omega_wheels_ref - wx.omega_wheels) / p.tau_omega

        delta_dot = (wx.delta_ref - wx.delta) / p.tau_delta

        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        if x.dim() == 1:
            return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot, torch.zeros_like(wx.front_friction), torch.zeros_like(wx.rear_friction), torch.zeros_like(wx.delta_ref), torch.zeros_like(wx.omega_wheels_ref_dot)])
        else:
            return torch.stack([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot, torch.zeros_like(wx.front_friction), torch.zeros_like(wx.rear_friction), torch.zeros_like(wx.delta_ref), torch.zeros_like(wx.omega_wheels_ref_dot)], dim=1)


class SingleTrackPacejkaModelNoDelay(torch.nn.Module):

    def __init__(self) -> None:
        super(SingleTrackPacejkaModelNoDelay, self).__init__()
        self.eps = 1e-6
        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()

        self.tire_model = PacejkaTireModel()

    def forward(self, t, x, p_vehicle, p_tire_front, p_tire_rear):
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = self.vehicle_parameters(p_vehicle)
        wx = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tire_front)
        wp_tire_r = self.tire_model_parameters(p_tire_rear)

        alpha_f = BaseTireModel.slip_angle_front_func(wx, p)
        alpha_r = BaseTireModel.slip_angle_rear_func(wx, p)
        slip_ratio_f = BaseTireModel.slip_ratio_front_func(wx, p)
        slip_ratio_r = BaseTireModel.slip_ratio_func(wx, p)

        # Tire Forces (per-axle friction)
        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r)
        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        Fy_f, Fx_f = Fy_f * wx.front_friction, Fx_f * wx.front_friction
        Fy_r, Fx_r = Fy_r * wx.rear_friction, Fx_r * wx.rear_friction
        self.last_tire_dynamics = {
            "Fx_f": Fx_f, "Fx_r": Fx_r, "Fy_f": Fy_f, "Fy_r": Fy_r,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "slip_ratio_f": slip_ratio_f, "slip_ratio_r": slip_ratio_r,
        }

        # Vehicle Dynamics

        # To remove delay
        omega_wheels_dot = torch.zeros_like(wx.omega_wheels)
        delta_dot = torch.zeros_like(wx.delta)

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta_ref) -
                               Fy_f * torch.sin(wx.delta_ref) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta_ref) +
                               Fy_r + Fy_f * torch.cos(wx.delta_ref) - p.m * wx.v_x * wx.r)

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta_ref) + Fy_f *
             torch.cos(wx.delta_ref)) * p.lf - Fy_r * p.lr)



        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        if x.dim() == 1:
            return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, wx.omega_wheels_ref_dot, torch.zeros_like(wx.omega_wheels), torch.zeros_like(wx.delta), torch.zeros_like(wx.front_friction), torch.zeros_like(wx.rear_friction), torch.zeros_like(wx.delta_ref), torch.zeros_like(wx.omega_wheels_ref_dot)])
        else:
            return torch.stack([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, wx.omega_wheels_ref_dot, torch.zeros_like(wx.omega_wheels), torch.zeros_like(wx.delta), torch.zeros_like(wx.front_friction), torch.zeros_like(wx.rear_friction), torch.zeros_like(wx.delta_ref), torch.zeros_like(wx.omega_wheels_ref_dot)], dim=1)


class SingleTrackPacejkaModelVelocityControls(torch.nn.Module):

    def __init__(self) -> None:
        super(SingleTrackPacejkaModelVelocityControls, self).__init__()
        self.eps = 1e-6
        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()

        self.tire_model = PacejkaTireModel()

    def forward(self, t, x, p_vehicle, p_tire_front, p_tire_rear):
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = self.vehicle_parameters(p_vehicle)
        wx = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tire_front)
        wp_tire_r = self.tire_model_parameters(p_tire_rear)

        alpha_f = BaseTireModel.slip_angle_front_func(wx, p)
        alpha_r = BaseTireModel.slip_angle_rear_func(wx, p)
        slip_ratio_f = BaseTireModel.slip_ratio_front_func(wx, p)
        slip_ratio_r = BaseTireModel.slip_ratio_func(wx, p)

        # Tire Forces (per-axle friction)
        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r)
        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        Fy_f, Fx_f = Fy_f * wx.front_friction, Fx_f * wx.front_friction
        Fy_r, Fx_r = Fy_r * wx.rear_friction, Fx_r * wx.rear_friction
        self.last_tire_dynamics = {
            "Fx_f": Fx_f, "Fx_r": Fx_r, "Fy_f": Fy_f, "Fy_r": Fy_r,
            "alpha_f": alpha_f, "alpha_r": alpha_r,
            "slip_ratio_f": slip_ratio_f, "slip_ratio_r": slip_ratio_r,
        }

        # Vehicle Dynamics

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - p.m * wx.v_x * wx.r)

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * p.lf - Fy_r * p.lr)

        omega_wheels_dot = (wx.omega_wheels_ref - wx.omega_wheels) / p.tau_omega

        delta_dot = (wx.delta_ref - wx.delta) / p.tau_delta

        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        if x.dim() == 1:
            return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, torch.zeros_like(wx.omega_wheels_ref_dot), delta_dot, torch.zeros_like(wx.front_friction), torch.zeros_like(wx.rear_friction), torch.zeros_like(wx.delta_ref), torch.zeros_like(wx.omega_wheels_ref_dot)])
        else:
            return torch.stack([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, torch.zeros_like(wx.omega_wheels_ref_dot), delta_dot, torch.zeros_like(wx.front_friction), torch.zeros_like(wx.rear_friction), torch.zeros_like(wx.delta_ref), torch.zeros_like(wx.omega_wheels_ref_dot)], dim=1)
