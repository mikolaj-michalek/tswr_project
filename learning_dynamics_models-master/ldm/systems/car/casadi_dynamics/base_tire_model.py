import casadi as ca

from ldm.systems.car.casadi_dynamics.state_wrapper import CasadiStateWrapper

# --- Base Tire Model ---
class CasadiBaseTireModel:
    def __init__(self):
        self.eps = 1e-6

    def lf(self, wp):
        return wp['L'] - wp['lr']

    def Fz_front(self, wp):
        return wp['m'] * wp['g'] * wp['lr'] / wp['L']

    def Fz_rear(self, wp):
        return wp['m'] * wp['g'] * self.lf(wp) / wp['L']

    def slip_angle_front_func(self, wx, wp):
        # alpha_f = atan((v_y + lf * r) / (v_x + eps)) - delta
        lf = self.lf(wp)
        return ca.atan((wx.v_y + lf * wx.r) / (wx.v_x + self.eps)) - wx.delta

    def slip_angle_rear_func(self, wx, wp):
        # alpha_r = atan((v_y - lr * r) / (v_x + eps))
        return ca.atan((wx.v_y - wp['lr'] * wx.r) / (wx.v_x + self.eps))

    def slip_ratio_front_func(self, wx, wp):
        v_front = wx.v_x * ca.cos(wx.delta) + (wx.v_y + wx.r * wp['lf']) * ca.sin(wx.delta)
        # Using fmax for stability (similar to torch.maximum)
        denom = ca.fmax(wx.omega_wheels_front, v_front) + self.eps
        return (wx.omega_wheels_front - v_front) / denom

    def slip_ratio_rear_func(self, wx, wp):
        v_rear = wx.v_x
        denom = ca.fmax(wx.omega_wheels_rear, v_rear) + self.eps
        return (wx.omega_wheels_rear - v_rear) / denom

    def tire_forces_model(self, slip_angle, slip_ratio, tire_params):
        raise NotImplementedError("Implement in subclass")

    def forward(self, x_u, wp_st, tire_params_f, tire_params_r):
        """
        x_u: concatenated state+control vector
        wp_st: dictionary of single track parameters
        tire_params_f: dictionary of front tire parameters
        tire_params_r: dictionary of rear tire parameters
        """
        wx = CasadiStateWrapper(x_u)
        
        alpha_f = self.slip_angle_front_func(wx, wp_st)
        kappa_f = self.slip_ratio_front_func(wx, wp_st)
        
        alpha_r = self.slip_angle_rear_func(wx, wp_st)
        kappa_r = self.slip_ratio_rear_func(wx, wp_st)
        
        # Calculate unit forces (friction coefficients)
        mu_xf, mu_yf = self.tire_forces_model(alpha_f, kappa_f, tire_params_f)
        mu_xr, mu_yr = self.tire_forces_model(alpha_r, kappa_r, tire_params_r)
        
        # Scale by normal load
        Fzf = self.Fz_front(wp_st)
        Fzr = self.Fz_rear(wp_st)
        
        Fxf = Fzf * mu_xf
        Fyf = Fzf * mu_yf
        Fxr = Fzr * mu_xr
        Fyr = Fzr * mu_yr
        
        # Return stacked forces [Fy_f, Fy_r, Fx_f, Fx_r] to match Torch output order
        return ca.vertcat(Fyf, Fyr, Fxf, Fxr)
