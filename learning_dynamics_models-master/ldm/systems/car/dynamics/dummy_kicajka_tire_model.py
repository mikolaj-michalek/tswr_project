import torch

from ldm.systems.car.dynamics import base_tire_model
from ldm.systems.car.dynamics.dummy_kicajka_params import DummyKicajkaParameters
from ldm.utils.bspline import BSpline


class DummyKicajkaTireModel(base_tire_model.BaseTireModel):
    def __init__(self, n, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = DummyKicajkaParameters(n)
        self.rear_tire_model_parameters = DummyKicajkaParameters(n)
        self.bspline = BSpline(n=n, d=3, num_T_pts=1024, name="kicajka")

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

        def compute_force(mod, scale, norm, res, cps):
            t = torch.clip(mod * scale, 0., 1.0)
            F = self.bspline.get_value(t, cps)[0]
            return (norm / res) * F
            
        
        Fx = compute_force(Sx_mod, wp.x_scale, Sx_norm, S_resultant, wp.cps_x)
        Fy = compute_force(Alpha_mod, wp.y_scale, Alpha_norm, S_resultant, wp.cps_y)
        #Ftot = compute_force(S_resultant, 1., 1., wp.xl_x, wp.yl_x, wp.xp_x, wp.yp_x, wp.xm_x, wp.ym_x)
        #Fx = (Sx_norm / S_resultant) * Ftot
        #Fy = (Alpha_norm / S_resultant) * Ftot

        #t = torch.linspace(0, 2.0, 1000)
        #t = torch.clip(t, 0., 1.0)
        #Ftot = compute_force(t, 1., 1., wp.xl_x, wp.yl_x, wp.xp_x, wp.yp_x, wp.xm_x, wp.ym_x)
        #Ftot = self.bspline.get_value(t, torch.tensor([0., 0.05, 0.2, 0.3, 0.4, 0.45, 0.45, 0.4, 0.35, 0.35]))

        #t = torch.clip(S_resultant, 0., 1.0)
        #Ftot = self.bspline.get_value(t, wp.cps)[0]
        #Fx = (Sx_norm / S_resultant) * Ftot
        #Fy = (Alpha_norm / S_resultant) * Ftot

        #import matplotlib.pyplot as plt
        #plt.plot(t.numpy(), Ftot.numpy())
        #plt.xlabel('Slip Ratio / Slip Angle')
        #plt.ylabel('Tire Force')
        #plt.title('Kicajka Tire Model Force Curve')
        #plt.grid()
        #plt.show()

        #slip_ratio = torch.clip(slip_ratio, -wp.xm_x, wp.xm_x)
        #slip_ratio_abs = torch.abs(slip_ratio)
        #Fx_lin = slip_ratio_abs * wp.yl_x / wp.xl_x
        #t_lin_peak = (slip_ratio_abs - wp.xl_x) / (wp.xp_x - wp.xl_x)
        #Fx_lin_peak = cubic_from_boundary(f0=wp.yl_x, df0=wp.yl_x / wp.xl_x, f1=wp.yp_x, df1=0., t=t_lin_peak)
        #t_peak_max = (slip_ratio_abs - wp.xp_x) / (wp.xm_x - wp.xp_x)
        #Fx_peak_max = cubic_from_boundary(f0=wp.yp_x, df0=0., f1=wp.ym_x, df1=0., t=t_peak_max)
        #Fx = torch.where(slip_ratio_abs < wp.xl_x, Fx_lin,
        #                 torch.where(slip_ratio_abs < wp.xp_x, Fx_lin_peak, Fx_peak_max))
        #Fx = (Sx_norm / S_resultant) * Fx
        
        return Fx, - Fy