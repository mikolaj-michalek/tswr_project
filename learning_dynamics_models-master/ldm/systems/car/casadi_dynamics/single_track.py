import os
import casadi as ca
import numpy as np
import torch

from ldm.systems.car.casadi_dynamics.kicajka4_tire_model import CasadiKicajka4TireModel
from ldm.systems.car.casadi_dynamics.pacejka_tire_model import CasadiPacejkaTireModel
from ldm.systems.car.casadi_dynamics.state_wrapper import CasadiStateWrapper
from ldm.systems.car.constants import MIN_VX
from ldm.systems.car.dynamics.single_track_params import DefaultSingleTrackParameters
from ldm.utils.loading import load_dynamics_model


class CasadiSingleTrack:
    def __init__(self, tire_model):
        self.tire_model = tire_model
        self.min_v_x = MIN_VX 
        self.vehicle_parameters = DefaultSingleTrackParameters()
        self.friction = 1.0

        self.x_symb = ca.MX.sym('x', 3)  # [v_x, v_y, r]
        self.u_symb = ca.MX.sym('u', 3)  # [omega_r, omega_f, delta]
        self.build_casadi_functions()

    def build_casadi_functions(self):
        x_dot, tire_forces = self.forward(self.x_symb, self.u_symb,
                                          self.vehicle_parameters.get_parameters_dict(),
                                          self.tire_model.front_tire_model_parameters.get_parameters_dict(),
                                          self.tire_model.rear_tire_model_parameters.get_parameters_dict(),
                                          self.friction)
        self.x_dot_func = ca.Function('x_dot_func', [self.x_symb, self.u_symb], [x_dot])
        self.tire_forces_func = ca.Function('tire_forces_func', [self.x_symb, self.u_symb], [tire_forces])

    def load_parameters_from_model(self, model_path):
        dynamics_model, _ = load_dynamics_model(model_path)
        self.vehicle_parameters = dynamics_model.vehicle_parameters
        self.tire_model.front_tire_model_parameters = dynamics_model.tire_model.front_tire_model_parameters
        self.tire_model.rear_tire_model_parameters = dynamics_model.tire_model.rear_tire_model_parameters
        self.friction = dynamics_model.log_friction.exp().item()
        self.build_casadi_functions()


    def forward(self, x, u, vehicle_params, tire_params_f, tire_params_r, friction_coeff):
        """
        x: [v_x, v_y, r]
        u: [omega_r, omega_f, delta]
        vehicle_params: dict
        tire_params_f: dict
        tire_params_r: dict
        friction_coeff: scalar
        """
        
        # Clamp state (v_x >= MIN_VX)
        # In CasADi, we build the graph with the clamped value
        v_x_clamped = ca.fmax(x[0], self.min_v_x)
        x_clamped = ca.vertcat(v_x_clamped, x[1], x[2])
        
        x_and_u = ca.vertcat(x_clamped, u)
        wx = CasadiStateWrapper(x_and_u)
        
        wp = vehicle_params
        
        # Get tire forces [Fy_f, Fy_r, Fx_f, Fx_r]
        raw_tire_forces = self.tire_model.forward(x_and_u, wp, tire_params_f, tire_params_r)
        
        # Apply global friction scaling
        tire_forces = raw_tire_forces * friction_coeff
        
        Fy_f = tire_forces[0]
        Fy_r = tire_forces[1]
        Fx_f = tire_forces[2]
        Fx_r = tire_forces[3]
        
        # Aerodynamic drag
        # torch.sign(wx.v_x) is approximated or used directly if v_x > 0 assumed
        # Since v_x clamped >= 2.0, sign is always 1.0
        sign_vx = 1.0 
        F_drag = wp['Cd0'] * sign_vx + \
                 wp['Cd1'] * wx.v_x + \
                 wp['Cd2'] * wx.v_x * wx.v_x
                 
        # Dynamics Equations
        # v_x_dot
        term1_x = Fx_r + Fx_f * ca.cos(wx.delta)
        term2_x = Fy_f * ca.sin(wx.delta)
        v_x_dot = (1.0 / wp['m']) * (term1_x - term2_x - F_drag + wp['m'] * wx.v_y * wx.r)
        
        # v_y_dot
        term1_y = Fx_f * ca.sin(wx.delta)
        term2_y = Fy_r + Fy_f * ca.cos(wx.delta)
        v_y_dot = (1.0 / wp['m']) * (term1_y + term2_y - wp['m'] * wx.v_x * wx.r)
        
        # r_dot
        term_r_f = (Fx_f * ca.sin(wx.delta) + Fy_f * ca.cos(wx.delta)) * self.tire_model.lf(wp)
        term_r_r = Fy_r * wp['lr']
        r_dot = (1.0 / wp['I_z']) * (term_r_f - term_r_r)
        
        x_dot = ca.vertcat(v_x_dot, v_y_dot, r_dot)
        
        return x_dot, tire_forces

if __name__ == "__main__":
    #model_path = os.path.join(os.path.dirname(__file__), "../../../../experiments/paper/f1tenth_new/prediction_models",
    #                          "pacejka_single_track_None_gc0.0001_lr0.0005_hf100_100_hb100_str2_euler_nn32_1.0_seed0.pt")
    model_path = os.path.join(os.path.dirname(__file__), "../../../../experiments/paper/f1tenth_new/prediction_models",
                              "kicajka4_single_track_None_gc0.0001_lr0.0005_hf100_100_hb100_str2_euler_nn32_1.0_n10_nup9_seed0.pt")
    dynamics_model, _ = load_dynamics_model(model_path)

    #tire_model = CasadiPacejkaTireModel()
    tire_model = CasadiKicajka4TireModel(n=10, n_up=9)
    dynamics = CasadiSingleTrack(tire_model)
    dynamics.load_parameters_from_model(model_path)

    x = np.array([4.0, 0.0, 0.1])  # Example state
    u = np.array([5.0, 5.0, 0.0])  # Example control
    x_dot_casadi = dynamics.x_dot_func(x, u)
    tire_forces_casadi = dynamics.tire_forces_func(x, u)

    # compare it with the PyTorch version to ensure consistency (not shown here, but should be done in testing)
    x_dot_torch, tire_forces_torch = dynamics_model(None, torch.tensor(x, dtype=torch.float32), torch.tensor(u, dtype=torch.float32))
    print("State derivatives casadi:", x_dot_casadi)
    print("PyTorch State derivatives:", x_dot_torch.detach().numpy())
    print("Tire forces casadi [Fy_f, Fy_r, Fx_f, Fx_r]:", tire_forces_casadi)
    print("PyTorch Tire forces [Fy_f, Fy_r, Fx_f, Fx_r]:", tire_forces_torch.detach().numpy())
    