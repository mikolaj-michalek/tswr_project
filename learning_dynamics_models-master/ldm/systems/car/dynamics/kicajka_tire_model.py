import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.kicajka_params import KicajkaParameters
from ldm.utils.bspline_torch import BSpline, _basis_functions


def _compute_forces_shared(bspline, t_f, cps_f, norm_xf, norm_yf,
                           t_r, cps_r, norm_xr, norm_yr, S_f, S_r):
    """Fuse front+rear BSpline evaluations into one _basis_functions call.

    Kicajka v1 uses a single shared CPS whose output is decomposed into
    Fx/Fy via the normalised-slip direction.
    """
    B = t_f.shape[0]
    N = _basis_functions(torch.cat([t_f, t_r]), bspline.knots, bspline.degree)
    Ftot_f = N[:B]  @ cps_f if cps_f.ndim == 1 else (N[:B]  * cps_f).sum(-1)
    Ftot_r = N[B:]  @ cps_r if cps_r.ndim == 1 else (N[B:]  * cps_r).sum(-1)
    Fx_f = (norm_xf / S_f) * Ftot_f
    Fx_r = (norm_xr / S_r) * Ftot_r
    return Fx_f, -(norm_yf / S_f) * Ftot_f, Fx_r, -(norm_yr / S_r) * Ftot_r


class KicajkaTireModel(BaseTireModel):
    def __init__(self, n, n_up, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = KicajkaParameters(n, n_up)
        self.rear_tire_model_parameters = KicajkaParameters(n, n_up)
        #self.bspline = BSpline(n=n+2, d=3, num_T_pts=1024, name="kicajka")
        self.bspline = torch.compile(BSpline(n_control=n+2, degree=3))

    #@staticmethod
    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        # alpha from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / torch.pi
        
        # Calculate normalized slip ratio and slip angle
        Sx_norm = slip_ratio / wp.x_norm
        Alpha_norm = slip_angle_rad / wp.y_norm
        
        # Compute the resultant slip
        S_resultant = torch.sqrt(Sx_norm**2 + Alpha_norm**2) + 1e-8
        
        # Find the modified slip factors
        Sx_mod = S_resultant * wp.x_norm
        Alpha_mod = S_resultant * wp.y_norm
        
        #Alpha_final = Alpha_mod 
        #Fy = ((Alpha_norm / S_resultant) * wp.Dy * torch.sin(wp.Cy * torch.atan((wp.By * Alpha_final) - 
        #    wp.Ey * -1.0 * (wp.By * Alpha_final - torch.atan(wp.By * Alpha_final))))) #+ wp.Svy
        
        ## Calculate the Longitudinal Force using Pacejka formula
        #Sx_final = Sx_mod #+ wp.Shx
        #Fx = ((Sx_norm / S_resultant) * wp.Dx * torch.sin(wp.Cx * torch.atan((wp.Bx * Sx_final) - 
        #    wp.Ex * -1.0 * (wp.Bx * Sx_final - torch.atan(wp.Bx * Sx_final))))) #+ wp.Svx

        #def compute_force(x, norm, res, xl, yl, xp, yp, xm, ym):
        #    x = torch.clip(x, -xm, xm)
        #    x_abs = torch.abs(x)
        #    F_lin = x_abs * yl / xl
        #    t_lin_peak = (x_abs - xl) / (xp - xl)
        #    F_lin_peak = cubic_from_boundary(f0=yl, df0=yl / xl, f1=yp, df1=0., t=t_lin_peak)
        #    t_peak_max = (x_abs - xp) / (xm - xp)
        #    F_peak_max = cubic_from_boundary(f0=yp, df0=0., f1=ym, df1=0., t=t_peak_max)
        #    F = torch.where(x_abs < xl, F_lin,
        #                    torch.where(x_abs < xp, F_lin_peak, F_peak_max))
        #    F = (norm / res) * F
        #    return F
            
        
        #Fx = compute_force(Sx_mod, Sx_norm, S_resultant, wp.xl_x, wp.yl_x, wp.xp_x, wp.yp_x, wp.xm_x, wp.ym_x)
        #Fy = compute_force(Alpha_mod, Alpha_norm, S_resultant, wp.xl_y, wp.yl_y, wp.xp_y, wp.yp_y, wp.xm_y, wp.ym_y)
        #Ftot = compute_force(S_resultant, 1., 1., wp.xl_x, wp.yl_x, wp.xp_x, wp.yp_x, wp.xm_x, wp.ym_x)
        #Fx = (Sx_norm / S_resultant) * Ftot
        #Fy = (Alpha_norm / S_resultant) * Ftot

        #t = torch.linspace(0, 2.0, 1000)
        #t = torch.clip(t, 0., 1.0)
        #Ftot = compute_force(t, 1., 1., wp.xl_x, wp.yl_x, wp.xp_x, wp.yp_x, wp.xm_x, wp.ym_x)
        #Ftot = self.bspline.get_value(t, torch.tensor([0., 0.05, 0.2, 0.3, 0.4, 0.45, 0.45, 0.4, 0.35, 0.35]))

        t = torch.clip(S_resultant, 0., 1.0)
        Ftot = self.bspline(t, wp.cps)
        Fx = (Sx_norm / S_resultant) * Ftot
        Fy = (Alpha_norm / S_resultant) * Ftot

        return Fx, - Fy

    def tire_forces(self, slip_ratio_front, slip_angle_front,
                    slip_ratio_rear, slip_angle_rear, wp_st):
        """Override to fuse front+rear BSpline evaluations into one call."""
        wp_f = self.front_tire_model_parameters()
        wp_r = self.rear_tire_model_parameters()

        # Front slip norms
        Sx_norm_f    = slip_ratio_front / wp_f.x_norm
        Alpha_norm_f = slip_angle_front / wp_f.y_norm
        S_f = torch.sqrt(Sx_norm_f**2 + Alpha_norm_f**2) + 1e-8

        # Rear slip norms
        Sx_norm_r    = slip_ratio_rear / wp_r.x_norm
        Alpha_norm_r = slip_angle_rear / wp_r.y_norm
        S_r = torch.sqrt(Sx_norm_r**2 + Alpha_norm_r**2) + 1e-8

        t_f = torch.clip(S_f, 0., 1.)
        t_r = torch.clip(S_r, 0., 1.)

        fric_xf, fric_yf, fric_xr, fric_yr = _compute_forces_shared(
            self.bspline,
            t_f, wp_f.cps, Sx_norm_f, Alpha_norm_f,
            t_r, wp_r.cps, Sx_norm_r, Alpha_norm_r,
            S_f, S_r,
        )

        Fxr = self.Fz_rear(wp_st)  * fric_xr
        Fyr = self.Fz_rear(wp_st)  * fric_yr
        Fxf = self.Fz_front(wp_st) * fric_xf
        Fyf = self.Fz_front(wp_st) * fric_yf
        return Fyf, Fyr, Fxf, Fxr