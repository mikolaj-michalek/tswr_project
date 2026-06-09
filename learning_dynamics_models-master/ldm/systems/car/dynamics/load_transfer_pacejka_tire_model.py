import torch

from ldm.systems.car.dynamics.pacejka_tire_model import PacejkaTireModel
from ldm.systems.car.dynamics.state_wrapper import ExtendedStateWrapper


class LoadTransferPacejkaTireModel(PacejkaTireModel):
    def forward(self, x, wp_st):

        wx = ExtendedStateWrapper(x)
        fric_xf, fric_yf = self.tire_forces_model(self.slip_angle_front_func(wx, wp_st),
                                                  self.slip_ratio_front_func(wx, wp_st), self.front_tire_model_parameters())
        
        Fxf = self.Fz_front(wp_st) * fric_xf
        Fyf = self.Fz_front(wp_st) * fric_yf

        F_aero = wp_st.Cl0 * torch.sign(wx.v_x) +\
            wp_st.Cl1 * wx.v_x +\
            wp_st.Cl2 * wx.v_x * wx.v_x
        #Fz_front = self.Fz_front(wp_st) + wx.dFz
        #Fz_rear = self.Fz_rear(wp_st) - wx.dFz
        Fz_front = self.Fz_front(wp_st) - wx.dFz + F_aero
        Fz_rear = self.Fz_rear(wp_st) + wx.dFz + F_aero
        
        Fxf = Fz_front * fric_xf
        Fyf = Fz_front * fric_yf
        
        fric_xr, fric_yr = self.tire_forces_model(self.slip_angle_rear_func(wx, wp_st),
                                                  self.slip_ratio_rear_func(wx, wp_st), self.rear_tire_model_parameters())

        
        Fxr = Fz_rear * fric_xr
        Fyr = Fz_rear * fric_yr
        
        friction = 1.0 # assume friction = 1.0, becuase it's already in Pacejka D params
        return torch.stack([Fyf, Fyr, Fxf, Fxr], dim=-1) * friction
        
        
