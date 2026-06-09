import casadi as ca
import numpy as np
import torch

from ldm.systems.car.casadi_dynamics.base_tire_model import CasadiBaseTireModel
from ldm.systems.car.casadi_dynamics.neural_utils import _torch_sequential_to_casadi
from ldm.systems.car.dynamics.neural_exp_tanh_tire_model import NeuralExpTanhTireModel


class CasadiNeuralExpTanhTireModel(CasadiBaseTireModel):
    def __init__(self):
        """
        Build CasADi symbolic functions from a *trained* torch model.

        Parameters
        ----------
        torch_model : NeuralExpTanhTireModel
            A (possibly trained) PyTorch neural-ExpTanh tire model whose
            weights will be baked into the CasADi expressions.
        """
        super().__init__()
        self.torch_model = None

        # ---- Build CasADi functions for the four sub-networks ----
        self._exptanh_params_front_fn = None
        self._exptanh_params_rear_fn = None

        self._s1s2_front_fn = None 
        self._s1s2_rear_fn = None

    def load_from_torch_model(self, torch_model: NeuralExpTanhTireModel):
        """Load parameters from a (possibly newly trained) torch model."""
        self.torch_model = torch_model
        self._exptanh_params_front_fn = self._build_mlp_function(
            torch_model.exptanh_params_front, 4, 'exptanh_f')
        self._exptanh_params_rear_fn = self._build_mlp_function(
            torch_model.exptanh_params_rear, 4, 'exptanh_r')
        self._s1s2_front_fn = self._build_mlp_function(
            torch_model.s1s2_front, 2, 's1s2_f')
        self._s1s2_rear_fn = self._build_mlp_function(
            torch_model.s1s2_rear, 2, 's1s2_r')

    @staticmethod
    def _build_mlp_function(seq, n_in, name):
        """Create a ca.Function wrapping a torch Sequential."""
        x_sym = ca.MX.sym('x', n_in, 1)
        y_sym = _torch_sequential_to_casadi(seq, x_sym)
        return ca.Function(name, [x_sym], [y_sym])

    # ------------------------------------------------------------------
    def compute_single_tire_forces(self, r, vx, vy, Fz,
                                   slip_angle, slip_ratio,
                                   exptanh_fn, s1s2_fn):
        """
        CasADi mirror of NeuralExpTanhTireModel.compute_single_tire_forces.
        exptanh_fn, s1s2_fn are ca.Function objects.
        """
        V = ca.sqrt(vx * vx + vy * vy)
        beta = ca.atan2(vy, vx)
        exptanh_input = ca.vertcat(r, V, beta, Fz)
        exptanh_out = exptanh_fn(exptanh_input)          # (5, 1)

        a1    = exptanh_out[0]
        a2    = exptanh_out[1]
        loga3 = exptanh_out[2]
        loga4 = exptanh_out[3]
        a5    = exptanh_out[4]
        a3 = ca.exp(loga3)
        a4 = ca.exp(loga4)

        k = ca.sqrt(ca.tan(slip_angle)**2 + slip_ratio**2)
        Ftot = a1 + a2 * ca.exp(-a3 * k) * ca.tanh(a4 * (k - a5))

        s1s2_input = ca.vertcat(slip_angle, slip_ratio)
        s1s2_out = s1s2_fn(s1s2_input)                   # (2, 1)
        s0 = s1s2_out[0]
        s1 = s1s2_out[1]
        s_norm = ca.sqrt(s0 * s0 + s1 * s1)

        Fy = (s0 * Ftot) / s_norm
        Fx = (s1 * Ftot) / s_norm
        return Fx, Fy

    # ------------------------------------------------------------------
    def forward(self, x_u, wp_st):
        """
        x_u:   CasADi symbolic state+control vector
        wp_st: dict of single-track vehicle parameters
        Returns: ca.vertcat(Fy_f, Fy_r, Fx_f, Fx_r)
        """
        from ldm.systems.car.casadi_dynamics.state_wrapper import CasadiStateWrapper
        wx = CasadiStateWrapper(x_u)

        alpha_f = self.slip_angle_front_func(wx, wp_st)
        kappa_f = self.slip_ratio_front_func(wx, wp_st)
        alpha_r = self.slip_angle_rear_func(wx, wp_st)
        kappa_r = self.slip_ratio_rear_func(wx, wp_st)

        Fz_front = self.Fz_front(wp_st)
        Fz_rear  = self.Fz_rear(wp_st)

        Fx_f, Fy_f = self.compute_single_tire_forces(
            wx.r, wx.v_x, wx.v_y, Fz_front,
            alpha_f, kappa_f,
            self._exptanh_params_front_fn, self._s1s2_front_fn,
        )
        Fx_r, Fy_r = self.compute_single_tire_forces(
            wx.r, wx.v_x, wx.v_y, Fz_rear,
            alpha_r, kappa_r,
            self._exptanh_params_rear_fn, self._s1s2_rear_fn,
        )

        return ca.vertcat(Fy_f, Fy_r, Fx_f, Fx_r)
