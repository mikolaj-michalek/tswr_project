import casadi as ca

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.casadi_dynamics.neural_utils import _torch_sequential_to_casadi
from ldm.systems.car.dynamics.neural_tire_model import NeuralTireModel


class CasadiNeuralTireModel(CasadiBaseTireModel):
    def __init__(self):
        """
        Build a CasADi symbolic tire-force model from a (possibly trained)
        NeuralTireModel whose MLP weights are baked into the expression.

        Parameters
        ----------
        torch_model : NeuralTireModel
            PyTorch neural tire model (weights are read, not modified).
        """
        super().__init__()

    def load_from_torch_model(self, torch_model: NeuralTireModel):
        self.input_type = torch_model.input_type
        self.n_in = torch_model.n_in

        # Build a ca.Function wrapping the MLP
        x_sym = ca.MX.sym('x', self.n_in, 1)
        y_sym = _torch_sequential_to_casadi(torch_model.mlp, x_sym)
        self._mlp_fn = ca.Function('neural_tire_mlp', [x_sym], [y_sym])

    def forward(self, x_u, wp_st):
        """
        x_u:   CasADi symbolic state+control vector
        wp_st: dict of single-track vehicle parameters
        Returns: ca.vertcat(Fy_f, Fy_r, Fx_f, Fx_r)
        """
        from ldm.systems.car.casadi_dynamics.state_wrapper import CasadiStateWrapper
        wx = CasadiStateWrapper(x_u)

        if self.input_type == "vinvariant":
            alpha_f = self.slip_angle_front_func(wx, wp_st)
            kappa_f = self.slip_ratio_front_func(wx, wp_st)
            alpha_r = self.slip_angle_rear_func(wx, wp_st)
            kappa_r = self.slip_ratio_rear_func(wx, wp_st)
            mlp_input = ca.vertcat(alpha_f, kappa_f, alpha_r, kappa_r)
        else:
            mlp_input = x_u[:self.n_in]

        scalers = self._mlp_fn(mlp_input)  # (4, 1)

        Fz_front = self.Fz_front(wp_st)
        Fz_rear = self.Fz_rear(wp_st)

        Fy_f = Fz_front * scalers[0]
        Fy_r = Fz_rear  * scalers[1]
        Fx_f = Fz_front * scalers[2]
        Fx_r = Fz_rear  * scalers[3]

        return ca.vertcat(Fy_f, Fy_r, Fx_f, Fx_r)
