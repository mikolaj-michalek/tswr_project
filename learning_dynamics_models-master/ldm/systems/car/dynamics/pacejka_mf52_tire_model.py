import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.pacejka_mf52_params import PacejkaMF52Parameters


class PacejkaMF52TireModel(BaseTireModel):
    """
    Pacejka Magic Formula 5.2 combined-slip tire model.

    Reference:
      Pacejka, H.B. (2006). Tyre and Vehicle Dynamics, 2nd ed.,
      Butterworth-Heinemann. Chapter 4, Section 4.3.2.

    Combined-slip formulation
    -------------------------
    Pure-slip forces are computed first using the standard MF:
        Fx0(κ),  Fy0(α)
    Then weighting functions G_xα and G_yκ scale each force according to
    the orthogonal slip component.

    G_xα (effect of slip angle on longitudinal force):
        α_S   = α_deg + SHxα
        Gxα0  = cos( Cxα · atan( Bxα·SHxα − Exα·(Bxα·SHxα − atan(Bxα·SHxα)) ) )
        Gxα   = cos( Cxα · atan( Bxα·α_S  − Exα·(Bxα·α_S  − atan(Bxα·α_S )) ) ) / Gxα0

    G_yκ (effect of slip ratio on lateral force):
        κ_S   = κ + SHyκ
        Gyκ0  = cos( Cyκ · atan( Byκ·SHyκ − Eyκ·(Byκ·SHyκ − atan(Byκ·SHyκ)) ) )
        Gyκ   = cos( Cyκ · atan( Byκ·κ_S  − Eyκ·(Byκ·κ_S  − atan(Byκ·κ_S )) ) ) / Gyκ0

    Key property: G = 1 at pure-slip conditions (both shifts = 0 ⟹ G0 = 1, G(0) = 1).

    Combined forces:
        Fx = Gxα · Fx0
        Fy = Gyκ · Fy0          (SVyκ ≈ 0 without camber)
    """

    def __init__(self, randomize_init: float = 0.0, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = PacejkaMF52Parameters(
            randomize_init=randomize_init
        )
        self.rear_tire_model_parameters = PacejkaMF52Parameters(
            randomize_init=randomize_init
        )

    @staticmethod
    def _magic(B, C, E, x):
        """Standard Pacejka Magic Formula argument evaluation."""
        Bx = B * x
        return C * torch.atan(Bx - E * (Bx - torch.atan(Bx)))

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        # Slip angle in degrees (consistent with existing codebase convention)
        alpha_deg = slip_angle_rad * 180.0 / torch.pi

        # ------------------------------------------------------------------
        # Pure-slip forces
        # ------------------------------------------------------------------
        # Shifted inputs
        alpha = alpha_deg + wp.Shy
        kappa = slip_ratio + wp.Shx

        Bx_a = wp.Bx * kappa
        Fx0 = wp.Dx * torch.sin(wp.Cx * torch.atan(Bx_a - wp.Ex * (Bx_a - torch.atan(Bx_a)))) + wp.Svx

        By_a = wp.By * alpha
        Fy0 = wp.Dy * torch.sin(wp.Cy * torch.atan(By_a - wp.Ey * (By_a - torch.atan(By_a)))) + wp.Svy

        # ------------------------------------------------------------------
        # MF 5.2 – G_xα: full MF shape (with E) in the weighting function
        # ------------------------------------------------------------------
        alpha_S  = alpha_deg + wp.SHxa
        Bxa_SH   = wp.Bxa * wp.SHxa
        Gxa_den  = torch.cos(wp.Cxa * torch.atan(Bxa_SH - wp.Exa * (Bxa_SH - torch.atan(Bxa_SH)))) + 1e-8
        Bxa_aS   = wp.Bxa * alpha_S
        Gxa_num  = torch.cos(wp.Cxa * torch.atan(Bxa_aS - wp.Exa * (Bxa_aS - torch.atan(Bxa_aS))))
        Gxa      = Gxa_num / Gxa_den

        # ------------------------------------------------------------------
        # MF 5.2 – G_yκ: full MF shape (with E) in the weighting function
        # ------------------------------------------------------------------
        kappa_S  = slip_ratio + wp.SHyk
        Byk_SH   = wp.Byk * wp.SHyk
        Gyk_den  = torch.cos(wp.Cyk * torch.atan(Byk_SH - wp.Eyk * (Byk_SH - torch.atan(Byk_SH)))) + 1e-8
        Byk_kS   = wp.Byk * kappa_S
        Gyk_num  = torch.cos(wp.Cyk * torch.atan(Byk_kS - wp.Eyk * (Byk_kS - torch.atan(Byk_kS))))
        Gyk      = Gyk_num / Gyk_den

        # ------------------------------------------------------------------
        # Combined forces
        # ------------------------------------------------------------------
        Fx = Gxa * Fx0
        Fy = Gyk * Fy0  # SVyκ = 0 (no camber)

        # BaseTireModel convention: return (Fx, -Fy)
        return Fx, -Fy
