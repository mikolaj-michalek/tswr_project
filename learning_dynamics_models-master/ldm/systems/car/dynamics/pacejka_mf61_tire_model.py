import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.pacejka_mf61_params import PacejkaMF61Parameters


class PacejkaMF61TireModel(BaseTireModel):
    """
    Pacejka Magic Formula 6.1 combined-slip tire model.

    Reference:
      Pacejka, H.B. (2012). Tyre and Vehicle Dynamics, 3rd ed.,
      Butterworth-Heinemann. Chapter 4, Eqs. 4.E70 – 4.E81.

    Structural differences from MF 5.2
    ------------------------------------
    1. G weighting functions use the SIMPLIFIED cos-atan form WITHOUT the
       curvature factor E:
           Gxα = cos(Cxα · atan(Bxα · α_S)) / cos(Cxα · atan(Bxα · SHxα))
           Gyκ = cos(Cyκ · atan(Byκ · κ_S)) / cos(Cyκ · atan(Byκ · SHyκ))
       (E is retained in the pure-slip formula only.)

    2. An additional lateral force term SV_yκ is included (MF 6.1, Eq. 4.E80):
           SVyκ = DVyk · sin(Cyκ · atan(Byκ · κ_S))
       Theoretically zero for zero camber angle, but kept as a learnable
       parameter to capture any data-driven bias from combined slip.

    Combined forces:
        Fx = Gxα · Fx0
        Fy = Gyκ · Fy0 + SVyκ
    """

    def __init__(self, randomize_init: float = 0.0, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = PacejkaMF61Parameters(
            randomize_init=randomize_init
        )
        self.rear_tire_model_parameters = PacejkaMF61Parameters(
            randomize_init=randomize_init
        )

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        # Slip angle in degrees (consistent with existing codebase convention)
        alpha_deg = slip_angle_rad * 180.0 / torch.pi

        # ------------------------------------------------------------------
        # Pure-slip forces (standard MF with E curvature factor)
        # ------------------------------------------------------------------
        alpha = alpha_deg + wp.Shy
        kappa = slip_ratio + wp.Shx

        Bx_a = wp.Bx * kappa
        Fx0  = wp.Dx * torch.sin(wp.Cx * torch.atan(Bx_a - wp.Ex * (Bx_a - torch.atan(Bx_a)))) + wp.Svx

        By_a = wp.By * alpha
        Fy0  = wp.Dy * torch.sin(wp.Cy * torch.atan(By_a - wp.Ey * (By_a - torch.atan(By_a)))) + wp.Svy

        # ------------------------------------------------------------------
        # MF 6.1 – G_xα: simplified cos(C·atan(B·α_S)) form, NO E in G
        # ------------------------------------------------------------------
        alpha_S = alpha_deg + wp.SHxa
        Gxa     = (torch.cos(wp.Cxa * torch.atan(wp.Bxa * alpha_S))
                   / (torch.cos(wp.Cxa * torch.atan(wp.Bxa * wp.SHxa)) + 1e-8))

        # ------------------------------------------------------------------
        # MF 6.1 – G_yκ: simplified cos(C·atan(B·κ_S)) form, NO E in G
        # ------------------------------------------------------------------
        kappa_S = slip_ratio + wp.SHyk
        Byk_kS  = wp.Byk * kappa_S
        Gyk     = (torch.cos(wp.Cyk * torch.atan(Byk_kS))
                   / (torch.cos(wp.Cyk * torch.atan(wp.Byk * wp.SHyk)) + 1e-8))

        # ------------------------------------------------------------------
        # MF 6.1 – SV_yκ: additional lateral force from combined slip
        # (Eq. 4.E80; DVyk = 0 initialisation → no effect at start of training)
        # ------------------------------------------------------------------
        SVyk = wp.DVyk * torch.sin(wp.Cyk * torch.atan(Byk_kS))

        # ------------------------------------------------------------------
        # Combined forces
        # ------------------------------------------------------------------
        Fx = Gxa * Fx0
        Fy = Gyk * Fy0 + SVyk

        # BaseTireModel convention: return (Fx, -Fy)
        return Fx, -Fy
