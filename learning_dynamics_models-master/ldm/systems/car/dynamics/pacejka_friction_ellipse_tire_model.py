import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.pacejka_offset_params import PacejkaOffsetParameters


class PacejkaFrictionEllipseTireModel(BaseTireModel):
    """
    Pacejka tire model with explicit friction ellipse constraint.

    Unlike the combined-slip approach in PacejkaOffsetTireModel (which blends
    forces by normalising the resultant slip vector), this model:

      1. Computes the pure longitudinal force Fx0 from the Pacejka longitudinal
         formula driven solely by the slip ratio (kappa).
      2. Computes the pure lateral force Fy0 from the Pacejka lateral formula
         driven solely by the slip angle (alpha).
      3. Projects the (Fx0, Fy0) vector onto the friction ellipse

             (Fx / Dx)^2 + (Fy / Dy)^2 <= 1

         by uniformly scaling both components when the vector exceeds the
         ellipse boundary.  Points already inside the ellipse are unchanged.

    References:
      [1] Bakker, E., Nyborg, L., Pacejka, H.B. (1987). Tyre modelling for use
          in vehicle dynamics studies.
      [2] Milliken & Milliken, Race Car Vehicle Dynamics. SAE, 1995.
    """

    def __init__(self, randomize_init: float = 0.0, *args, **kwargs) -> None:
        super().__init__()
        self.front_tire_model_parameters = PacejkaOffsetParameters(
            randomize_init=randomize_init
        )
        self.rear_tire_model_parameters = PacejkaOffsetParameters(
            randomize_init=randomize_init
        )

    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        # Convert slip angle from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / torch.pi

        # Apply horizontal input offsets (Shy shifts the slip angle,
        # Shx shifts the slip ratio)
        alpha = slip_angle_deg + wp.Shy   # shifted slip angle [deg]
        kappa = slip_ratio + wp.Shx       # shifted slip ratio  [-]

        # ------------------------------------------------------------------
        # Pure lateral force – Pacejka formula evaluated at slip angle only
        # ------------------------------------------------------------------
        Fy0 = (
            wp.Dy
            * torch.sin(
                wp.Cy
                * torch.atan(
                    wp.By * alpha
                    - wp.Ey * (wp.By * alpha - torch.atan(wp.By * alpha))
                )
            )
            + wp.Svy
        )

        # ------------------------------------------------------------------
        # Pure longitudinal force – Pacejka formula evaluated at slip ratio only
        # ------------------------------------------------------------------
        Fx0 = (
            wp.Dx
            * torch.sin(
                wp.Cx
                * torch.atan(
                    wp.Bx * kappa
                    - wp.Ex * (wp.Bx * kappa - torch.atan(wp.Bx * kappa))
                )
            )
            + wp.Svx
        )

        # ------------------------------------------------------------------
        # Friction ellipse constraint
        # Normalise each component by its peak value (Dx, Dy) and compute
        # the ellipse utilisation factor:
        #   ellipse_norm = sqrt( (Fx0/Dx)^2 + (Fy0/Dy)^2 )
        # A value > 1 means the combined force exceeds the ellipse, so both
        # components are scaled down by 1/ellipse_norm.
        # ------------------------------------------------------------------
        ellipse_norm = torch.sqrt(
            (Fx0 / (wp.Dx + 1e-8)) ** 2
            + (Fy0 / (wp.Dy + 1e-8)) ** 2
            + 1e-8
        )

        # scale = min(1, 1/ellipse_norm):  identity inside, projection outside
        scale = torch.clamp(1.0 / ellipse_norm, max=1.0)

        Fx = Fx0 * scale
        Fy = Fy0 * scale

        # BaseTireModel convention: return (Fx, -Fy) so that positive Fy0
        # (force opposing positive slip angle) is negated to match the
        # coordinate frame used by SingleTrack.
        return Fx, -Fy
