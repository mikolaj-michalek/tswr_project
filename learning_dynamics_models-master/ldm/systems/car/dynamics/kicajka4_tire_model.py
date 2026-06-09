from time import perf_counter
import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.kicajka4_params import Kicajka4Parameters
from ldm.systems.car.dynamics.kicajka_utils import compute_force, compute_forces_quad
#from ldm.utils.bspline import BSpline
from ldm.utils.bspline_torch import BSpline


class Kicajka4TireModel(BaseTireModel):
    def __init__(self, n, n_up, randomize_init=0., *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = Kicajka4Parameters(n, n_up, randomize_init)
        self.rear_tire_model_parameters = Kicajka4Parameters(n, n_up, randomize_init)
        #self.bspline = BSpline(n=n+2, d=3, num_T_pts=1024, name="kicajka")
        self.bspline = torch.compile(BSpline(n_control=n+2, degree=3))

    #@staticmethod
    def tire_forces_model(self, slip_angle_rad, slip_ratio, wp):
        # Calculate normalized slip ratio and slip angle
        #times = []
        #times.append(perf_counter())
        Sx_norm = (slip_ratio + wp.sr_offset) / wp.x_norm
        Alpha_norm = (slip_angle_rad + wp.sa_offset) / wp.y_norm
        
        # Compute the resultant slip
        S_resultant = torch.sqrt(Sx_norm**2 + Alpha_norm**2) + 1e-8
        
        # Find the modified slip factors
        Sx_mod = S_resultant * wp.x_norm
        Alpha_mod = S_resultant * wp.y_norm
        #times.append(perf_counter())

        Fx = compute_force(self.bspline, Sx_mod, wp.x_scale, Sx_norm, S_resultant, wp.cps_x)
        #times.append(perf_counter())
        Fy = compute_force(self.bspline, Alpha_mod, wp.y_scale, Alpha_norm, S_resultant, wp.cps_y)
        #times.append(perf_counter())
        #print("INSIDE TIRE FORCES TIMES: ", [f"{(times[i+1]-times[i])*1000:.1f}ms" for i in range(len(times)-1)])
        return Fx, - Fy

    def tire_forces(self, slip_ratio_front, slip_angle_front,
                    slip_ratio_rear, slip_angle_rear, wp_st):
        """Override to fuse all four BSpline evaluations into one call."""
        wp_f = self.front_tire_model_parameters()
        wp_r = self.rear_tire_model_parameters()

        # Front slip norms
        Sx_norm_f    = (slip_ratio_front  + wp_f.sr_offset) / wp_f.x_norm
        Alpha_norm_f = (slip_angle_front  + wp_f.sa_offset) / wp_f.y_norm
        S_f = torch.sqrt(Sx_norm_f**2 + Alpha_norm_f**2) + 1e-8

        # Rear slip norms
        Sx_norm_r    = (slip_ratio_rear   + wp_r.sr_offset) / wp_r.x_norm
        Alpha_norm_r = (slip_angle_rear   + wp_r.sa_offset) / wp_r.y_norm
        S_r = torch.sqrt(Sx_norm_r**2 + Alpha_norm_r**2) + 1e-8

        # Pre-clip t values: t = clip(S * x_norm * x_scale, 0, 1)
        t_xf = torch.clip(S_f * wp_f.x_norm * wp_f.x_scale, 0., 1.)
        t_yf = torch.clip(S_f * wp_f.y_norm * wp_f.y_scale, 0., 1.)
        t_xr = torch.clip(S_r * wp_r.x_norm * wp_r.x_scale, 0., 1.)
        t_yr = torch.clip(S_r * wp_r.y_norm * wp_r.y_scale, 0., 1.)

        fric_xf, fric_yf, fric_xr, fric_yr = compute_forces_quad(
            self.bspline,
            t_xf, wp_f.cps_x, Sx_norm_f,
            t_yf, wp_f.cps_y, Alpha_norm_f,
            t_xr, wp_r.cps_x, Sx_norm_r,
            t_yr, wp_r.cps_y, Alpha_norm_r,
            S_f, S_r,
        )

        Fxr = self.Fz_rear(wp_st)  * fric_xr
        Fyr = self.Fz_rear(wp_st)  * fric_yr
        Fxf = self.Fz_front(wp_st) * fric_xf
        Fyf = self.Fz_front(wp_st) * fric_yf
        return Fyf, Fyr, Fxf, Fxr