import math
import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.state_wrapper import StateWrapper

class NeuralExpTanhTireModel(BaseTireModel):
    def __init__(self, nn=16, *args, **kwargs) -> None:
        super().__init__()
        #act = torch.nn.ReLU()
        act = torch.nn.Tanh()
        #act = torch.nn.ELU()
        self.exptanh_params_front = torch.nn.Sequential(
            torch.nn.Linear(4, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, 5),
        )
        self.s1s2_front = torch.nn.Sequential(
            torch.nn.Linear(2, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, 2),
        )
        self.exptanh_params_rear = torch.nn.Sequential(
            torch.nn.Linear(4, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, 5),
        )
        self.s1s2_rear = torch.nn.Sequential(
            torch.nn.Linear(2, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, 2),
        )

        # Initialise last-layer biases to the default ExpTanh parameter values:
        #   [a1=0, a2=1, log_a3=log(0.1), log_a4=log(5), a5=0]
        _exptanh_bias = torch.tensor([0.0, 1.0, math.log(0.1), math.log(5.0), 0.0])
        #   [s1=1, s2=1]
        _s1s2_bias = torch.tensor([1.0, 1.0])
        with torch.no_grad():
            self.exptanh_params_front[-1].bias.copy_(_exptanh_bias)
            self.exptanh_params_rear[-1].bias.copy_(_exptanh_bias)
            self.s1s2_front[-1].bias.copy_(_s1s2_bias)
            self.s1s2_rear[-1].bias.copy_(_s1s2_bias)


    def compute_single_tire_forces(self, r, vx, vy, Fz, slip_angle, slip_ratio, exptanh_fn, s1s2_fn):
        V = torch.sqrt(vx * vx + vy * vy)
        beta = torch.atan2(vy, vx)
        exptanh_params_input = torch.stack([r, V, beta, Fz * torch.ones_like(V)], dim=-1)
        exptanh_params = exptanh_fn(exptanh_params_input)
        s1s2_input = torch.stack([slip_angle, slip_ratio], dim=-1)
        s1s2 = s1s2_fn(s1s2_input)
        a1, a2, loga3, loga4, a5 = exptanh_params.unbind(-1)
        a3 = torch.exp(loga3)
        a4 = torch.exp(loga4)
        k = torch.sqrt(torch.tan(slip_angle)**2 + slip_ratio**2)
        Ftot = a1 + a2 * torch.exp(-a3 * k) * torch.tanh(a4 * (k - a5))
        Fy = (s1s2[..., 0] * Ftot) / torch.sqrt(s1s2.square().sum(-1)) 
        Fx = (s1s2[..., 1] * Ftot) / torch.sqrt(s1s2.square().sum(-1))
        return Fx, Fy

    def forward(self, x, wp_st):

        wx = StateWrapper(x)
        slip_angle_front = self.slip_angle_front_func(wx, wp_st)
        slip_ratio_front = self.slip_ratio_front_func(wx, wp_st)
        slip_angle_rear = self.slip_angle_rear_func(wx, wp_st)
        slip_ratio_rear = self.slip_ratio_rear_func(wx, wp_st)

        if hasattr(wx, 'dFz'):
            # with load transfer
            #Fz_front = self.Fz_front(wp_st) + wx.dFz
            #Fz_rear = self.Fz_rear(wp_st) - wx.dFz
            F_aero = wp_st.Cl0 * torch.sign(wx.v_x) +\
                wp_st.Cl1 * wx.v_x +\
                wp_st.Cl2 * wx.v_x * wx.v_x
            Fz_front = self.Fz_front(wp_st) - wx.dFz + F_aero
            Fz_rear = self.Fz_rear(wp_st) + wx.dFz + F_aero
        else:
            Fz_front = self.Fz_front(wp_st)
            Fz_rear = self.Fz_rear(wp_st)

        Fx_f, Fy_f = self.compute_single_tire_forces(
            wx.r, wx.v_x, wx.v_y, Fz_front, slip_angle_front, slip_ratio_front, self.exptanh_params_front, self.s1s2_front
        )
        Fx_r, Fy_r = self.compute_single_tire_forces(
            wx.r, wx.v_x, wx.v_y, Fz_rear, slip_angle_rear, slip_ratio_rear, self.exptanh_params_rear, self.s1s2_rear
        )
        forces = torch.stack([Fy_f, Fy_r, Fx_f, Fx_r], dim=-1)
        return forces