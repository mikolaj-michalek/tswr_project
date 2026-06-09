import casadi as ca

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.dynamics.exp_tanh_tire_model import ExpTanhTireModel


class CasadiExpTanhTireModel(CasadiBaseTireModel):
    def __init__(self):
        super().__init__()
        # Keep a torch version around so we can load/sync parameters
        self._torch_model = ExpTanhTireModel()

    @staticmethod
    def compute_single_tire_forces(Fz, slip_angle, slip_ratio,
                                   a1, a2, a3, a4, a5, s1, s2):
        """
        CasADi version of ExpTanhTireModel.compute_single_tire_forces.
        a3 and a4 are expected *already exponentiated* (i.e. positive values).
        Returns: Fx, Fy (absolute forces, not unit/normalised).
        """
        sa_norm = slip_angle / s1
        sr_norm = slip_ratio / s2
        k = ca.sqrt(sa_norm**2 + sr_norm**2)
        Ftot = Fz * (a1 + a2 * ca.exp(-a3 * k) * ca.tanh(a4 * (k - a5)))
        Fy = (sa_norm * Ftot) / k
        Fx = (sr_norm * Ftot) / k
        return Fx, Fy

    def forward(self, x_u, wp_st, tp_f, tp_r):
        """
        x_u:   concatenated state+control CasADi symbolic vector
        wp_st: dict of single-track vehicle parameters
        tp_f:  dict of front tire ExpTanh parameters
               {'a1', 'a2', 'a3', 'a4', 'a5', 's1', 's2'}
               (a3, a4 are the *exponentiated* values)
        tp_r:  dict of rear tire ExpTanh parameters (same keys)
        """
        from ldm.systems.car.casadi_dynamics.state_wrapper import CasadiStateWrapper
        wx = CasadiStateWrapper(x_u)

        alpha_f = self.slip_angle_front_func(wx, wp_st)
        kappa_f = self.slip_ratio_front_func(wx, wp_st)
        alpha_r = self.slip_angle_rear_func(wx, wp_st)
        kappa_r = self.slip_ratio_rear_func(wx, wp_st)

        Fz_front = self.Fz_front(wp_st)
        Fz_rear = self.Fz_rear(wp_st)

        Fx_f, Fy_f = self.compute_single_tire_forces(
            Fz_front, alpha_f, kappa_f,
            tp_f['a1'], tp_f['a2'], tp_f['a3'], tp_f['a4'], tp_f['a5'],
            tp_f['s1'], tp_f['s2'],
        )
        Fx_r, Fy_r = self.compute_single_tire_forces(
            Fz_rear, alpha_r, kappa_r,
            tp_r['a1'], tp_r['a2'], tp_r['a3'], tp_r['a4'], tp_r['a5'],
            tp_r['s1'], tp_r['s2'],
        )

        # Return stacked forces [Fy_f, Fy_r, Fx_f, Fx_r] to match Torch output order
        return ca.vertcat(Fy_f, Fy_r, Fx_f, Fx_r)

    # ------------------------------------------------------------------
    # Helpers to extract parameter dicts from a trained torch model
    # ------------------------------------------------------------------
    @staticmethod
    def torch_params_to_dict(torch_model, prefix):
        """
        Extract the ExpTanh parameters for one axle from the torch model.

        prefix: 'f' or 'r'
        Returns a plain dict with keys a1..a5, s1, s2 (a3/a4 exponentiated).
        """
        return {
            'a1': getattr(torch_model, f'a1_{prefix}').detach().item(),
            'a2': getattr(torch_model, f'a2_{prefix}').detach().item(),
            'a3': getattr(torch_model, f'log_a3_{prefix}').exp().detach().item(),
            'a4': getattr(torch_model, f'log_a4_{prefix}').exp().detach().item(),
            'a5': getattr(torch_model, f'a5_{prefix}').detach().item(),
            's1': getattr(torch_model, f's1_{prefix}').detach().item(),
            's2': getattr(torch_model, f's2_{prefix}').detach().item(),
        }
