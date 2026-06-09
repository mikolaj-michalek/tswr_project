import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.state_wrapper import StateWrapper


class ExpTanhTireModel(BaseTireModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.a1_f = torch.nn.Parameter(torch.tensor([0.0]))
        self.a2_f = torch.nn.Parameter(torch.tensor([1.0]))
        self.log_a3_f = torch.nn.Parameter(torch.tensor([0.1]).log())
        self.log_a4_f = torch.nn.Parameter(torch.tensor([5.0]).log())
        self.a5_f = torch.nn.Parameter(torch.tensor([0.0]))
        self.s1_f = torch.nn.Parameter(torch.tensor([1.0]))
        self.s2_f = torch.nn.Parameter(torch.tensor([1.0]))

        self.a1_r = torch.nn.Parameter(torch.tensor([0.0]))
        self.a2_r = torch.nn.Parameter(torch.tensor([1.0]))
        self.log_a3_r = torch.nn.Parameter(torch.tensor([0.1]).log())
        self.log_a4_r = torch.nn.Parameter(torch.tensor([5.0]).log())
        self.a5_r = torch.nn.Parameter(torch.tensor([0.0]))
        self.s1_r = torch.nn.Parameter(torch.tensor([1.0]))
        self.s2_r = torch.nn.Parameter(torch.tensor([1.0]))


    def compute_single_tire_forces(self, Fz, slip_angle, slip_ratio, a1, a2, log_a3, log_a4, a5, s1, s2):
        # My ExpTanh tire model
        a3 = torch.exp(log_a3)
        a4 = torch.exp(log_a4)
        sa_norm = slip_angle / s1
        sr_norm = slip_ratio / s2
        k = torch.sqrt(sa_norm**2 + sr_norm**2)
        Ftot = Fz * (a1 + a2 * torch.exp(-a3 * k) * torch.tanh(a4 * (k - a5)))
        Fy = (sa_norm * Ftot) / k
        Fx = (sr_norm * Ftot) / k

        # Classic ExpTanh tire model 
        #a3 = torch.exp(log_a3)
        #a4 = torch.exp(log_a4)
        #k = torch.sqrt(torch.tan(slip_angle)**2 + slip_ratio**2)
        #Ftot = Fz * (a1 + a2 * torch.exp(-a3 * k) * torch.tanh(a4 * (k - a5)))
        #sa_norm = slip_angle / s1
        #sr_norm = slip_ratio / s2
        #denom = torch.sqrt(sa_norm**2 + sr_norm**2) 
        #Fy = (sa_norm * Ftot) / denom
        #Fx = (sr_norm * Ftot) / denom
        return Fx, Fy

    def tire_forces(self, slip_ratio_front, slip_angle_front,
                      slip_ratio_rear, slip_angle_rear, wp_st):
        Fz_front = self.Fz_front(wp_st)
        Fz_rear = self.Fz_rear(wp_st)
        Fx_f, Fy_f = self.compute_single_tire_forces(
            Fz_front, slip_angle_front, slip_ratio_front, self.a1_f, self.a2_f,
            self.log_a3_f, self.log_a4_f, self.a5_f, self.s1_f, self.s2_f
        )
        Fx_r, Fy_r = self.compute_single_tire_forces(
            Fz_rear, slip_angle_rear, slip_ratio_rear, self.a1_r, self.a2_r,
            self.log_a3_r, self.log_a4_r, self.a5_r, self.s1_r, self.s2_r
        )
        return Fy_f, Fy_r, Fx_f, Fx_r

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
            Fz_front, slip_angle_front, slip_ratio_front, self.a1_f, self.a2_f,
            self.log_a3_f, self.log_a4_f, self.a5_f, self.s1_f, self.s2_f
        )
        Fx_r, Fy_r = self.compute_single_tire_forces(
            Fz_rear, slip_angle_rear, slip_ratio_rear, self.a1_r, self.a2_r,
            self.log_a3_r, self.log_a4_r, self.a5_r, self.s1_r, self.s2_r
        )
        forces = torch.stack([Fy_f, Fy_r, Fx_f, Fx_r], dim=-1)
        return forces
        
