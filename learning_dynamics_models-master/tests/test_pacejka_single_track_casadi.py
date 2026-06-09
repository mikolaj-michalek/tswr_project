import torch
import numpy as np
import casadi as ca
import unittest

# Import Torch modules
from ldm.systems.car.casadi_dynamics.pacejka_tire_model import CasadiPacejkaTireModel
from ldm.systems.car.casadi_dynamics.single_track import CasadiSingleTrack
from ldm.systems.car.casadi_dynamics.state_wrapper import CasadiStateWrapper
from ldm.systems.car.dynamics.pacejka_params import PacejkaParameters
from ldm.systems.car.dynamics.pacejka_tire_model import PacejkaTireModel
from ldm.systems.car.dynamics.single_track_params import DefaultSingleTrackParameters
from ldm.systems.car.dynamics.single_track import SingleTrack
from ldm.systems.car.dynamics.state_wrapper import StateWrapper

class TestDynamicsEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        
        # 1. Initialize Torch Model
        self.vehicle_params_torch = DefaultSingleTrackParameters()
        self.tire_model_torch = PacejkaTireModel()
        self.model_torch = SingleTrack(self.vehicle_params_torch, self.tire_model_torch)
        
        # 2. Initialize Casadi Model
        self.casadi_tire = CasadiPacejkaTireModel()
        self.model_casadi = CasadiSingleTrack(self.casadi_tire)

    def extract_params_as_dict(self):
        """Helper to evaluate Torch parameters (exp conversions) and export as dict for CasADi"""
        # Vehicle Params
        vp_obj = self.vehicle_params_torch()
        vp_dict = {k: getattr(vp_obj, k).detach().item() for k in vp_obj._fields}
        
        # Tire Params (Front and Rear are initialized same in PacejkaTireModel)
        # Note: In the torch code provided, front/rear use separate PacejkaParameters instances
        tp_f_obj = self.tire_model_torch.front_tire_model_parameters()
        tp_f_dict = {k: getattr(tp_f_obj, k).detach().item() for k in tp_f_obj._fields}
        
        tp_r_obj = self.tire_model_torch.rear_tire_model_parameters()
        tp_r_dict = {k: getattr(tp_r_obj, k).detach().item() for k in tp_r_obj._fields}
        
        # Global Friction
        friction = self.model_torch.log_friction.exp().detach().item()
        
        return vp_dict, tp_f_dict, tp_r_dict, friction

    def test_forward_pass(self):
        # --- Prepare Data ---
        # State: [v_x, v_y, r]
        # Control: [omega_r, omega_f, delta]
        # Ensure v_x is large enough to avoid min_state clamping logic differences causing divergence immediately
        # though the logic is replicated in CasADi.
        batch_size = 1
        x_torch = torch.tensor([[10.0, 0.5, 0.2]], dtype=torch.float32)
        u_torch = torch.tensor([[12.0, 12.0, 0.05]], dtype=torch.float32)
        
        # --- Torch Forward ---
        # time t is not used in the provided forward code
        dx_torch, forces_torch = self.model_torch(0.0, x_torch, u_torch)
        
        dx_torch_np = dx_torch.detach().numpy().flatten()
        forces_torch_np = forces_torch.detach().numpy().flatten()
        
        # --- Casadi Forward ---
        vp, tp_f, tp_r, fric = self.extract_params_as_dict()
        
        # Define Casadi Symbolics
        x_sym = ca.MX.sym('x', 3)
        u_sym = ca.MX.sym('u', 3)
        
        # Build graph
        dx_sym, forces_sym = self.model_casadi.forward(x_sym, u_sym, vp, tp_f, tp_r, fric)
        
        # Create function
        f_casadi = ca.Function('f', [x_sym, u_sym], [dx_sym, forces_sym])
        
        # Evaluate
        x_in = x_torch.numpy().flatten()
        u_in = u_torch.numpy().flatten()
        
        res = f_casadi(x_in, u_in)
        dx_casadi_np = np.array(res[0]).flatten()
        forces_casadi_np = np.array(res[1]).flatten()
        
        # --- Comparison ---
        print("\n--- State Derivatives (v_x_dot, v_y_dot, r_dot) ---")
        print(f"Torch:  {dx_torch_np}")
        print(f"Casadi: {dx_casadi_np}")
        
        print("\n--- Tire Forces (Fy_f, Fy_r, Fx_f, Fx_r) ---")
        print(f"Torch:  {forces_torch_np}")
        print(f"Casadi: {forces_casadi_np}")

        x_u = ca.vertcat(x_in, u_in)
        wx = CasadiStateWrapper(x_u)
        sa_front = self.casadi_tire.slip_angle_front_func(wx, vp)
        sa_rear = self.casadi_tire.slip_angle_rear_func(wx, vp)
        sr_front = self.casadi_tire.slip_ratio_front_func(wx, vp)
        sr_rear = self.casadi_tire.slip_ratio_rear_func(wx, vp)

        wx_torch = StateWrapper(torch.cat([x_torch, u_torch], dim=-1))
        wp = self.vehicle_params_torch()
        sa_front_torch = self.tire_model_torch.slip_angle_front_func(wx_torch, wp)
        sa_rear_torch = self.tire_model_torch.slip_angle_rear_func(wx_torch, wp)
        sr_front_torch = self.tire_model_torch.slip_ratio_front_func(wx_torch, wp)
        sr_rear_torch = self.tire_model_torch.slip_ratio_rear_func(wx_torch, wp)

        np.testing.assert_allclose(sa_front_torch.item(), float(sa_front), rtol=1e-5, atol=1e-6, 
                                   err_msg="Front slip angle does not match")
        np.testing.assert_allclose(sa_rear_torch.item(), float(sa_rear), rtol=1e-5, atol=1e-6, 
                                   err_msg="Rear slip angle does not match")
        np.testing.assert_allclose(sr_front_torch.item(), float(sr_front), rtol=1e-5, atol=1e-6, 
                                   err_msg="Front slip ratio does not match")
        np.testing.assert_allclose(sr_rear_torch.item(), float(sr_rear), rtol=1e-5, atol=1e-6, 
                                   err_msg="Rear slip ratio does not match")

        # Assertions
        # Tolerance: floating point differences between C++ (CasADi) and Torch can occur
        np.testing.assert_allclose(dx_torch_np, dx_casadi_np, rtol=1e-5, atol=1e-6, 
                                   err_msg="State derivatives do not match")
        np.testing.assert_allclose(forces_torch_np, forces_casadi_np, rtol=1e-5, atol=1e-6, 
                                   err_msg="Tire forces do not match")

    def test_clamping_logic(self):
        """Test if the v_x clamping works identically in both"""
        # v_x below 2.0
        x_torch = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32) 
        u_torch = torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float32)
        
        dx_torch, _ = self.model_torch(0.0, x_torch, u_torch)
        dx_torch_np = dx_torch.detach().numpy().flatten()
        
        vp, tp_f, tp_r, fric = self.extract_params_as_dict()
        
        # Casadi
        x_sym = ca.MX.sym('x', 3)
        u_sym = ca.MX.sym('u', 3)
        dx_sym, _ = self.model_casadi.forward(x_sym, u_sym, vp, tp_f, tp_r, fric)
        f_casadi = ca.Function('f_clamp', [x_sym, u_sym], [dx_sym])
        
        dx_casadi_np = np.array(f_casadi(x_torch.numpy().flatten(), u_torch.numpy().flatten())).flatten()
        
        np.testing.assert_allclose(dx_torch_np, dx_casadi_np, rtol=1e-5, atol=1e-6,
                                   err_msg="Clamping logic mismatch")

if __name__ == '__main__':
    unittest.main()