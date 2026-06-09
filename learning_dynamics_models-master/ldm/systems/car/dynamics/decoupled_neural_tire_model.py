import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.state_wrapper import StateWrapper


class DecoupledNeuralTireModel(BaseTireModel):
    def __init__(self, nn=32, *args, **kwargs) -> None:
        super().__init__()
        act = torch.nn.ReLU()
        #act = torch.nn.Tanh()
        #act = torch.nn.ELU()
        self.mlp_front = torch.nn.Sequential(
            torch.nn.Linear(2, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, 2),
        )

        self.mlp_rear = torch.nn.Sequential(
            torch.nn.Linear(2, nn),
            act,
            torch.nn.Linear(nn, nn),
            act,
            torch.nn.Linear(nn, 2),
        )

    def _tire_forces(self, scalers_front, scalers_rear, Fz_front, Fz_rear):
        scalers = torch.stack([scalers_front[..., 0], scalers_rear[..., 0],
                               scalers_front[..., 1], scalers_rear[..., 1]], dim=-1)
        forces = torch.stack([
            Fz_front * scalers[..., 0],
            Fz_rear * scalers[..., 1],
            Fz_front * scalers[..., 2],
            Fz_rear * scalers[..., 3],
        ], dim=-1)
        return forces

    def tire_forces(self, slip_ratio_front, slip_angle_front,
                          slip_ratio_rear, slip_angle_rear, wp_st):
        input_front = torch.stack([slip_angle_front, slip_ratio_front], dim=-1)
        scalers_front = self.mlp_front(input_front)
        input_rear = torch.stack([slip_angle_rear, slip_ratio_rear], dim=-1)
        scalers_rear = self.mlp_rear(input_rear)
        Fz_front = self.Fz_front(wp_st)
        Fz_rear = self.Fz_rear(wp_st)
        forces = self._tire_forces(scalers_front, scalers_rear, Fz_front, Fz_rear)
        return forces.unbind(-1)

    def forward(self, x, wp_st):

        wx = StateWrapper(x)
        slip_angle_front = self.slip_angle_front_func(wx, wp_st)
        slip_ratio_front = self.slip_ratio_front_func(wx, wp_st)
        slip_angle_rear = self.slip_angle_rear_func(wx, wp_st)
        slip_ratio_rear = self.slip_ratio_rear_func(wx, wp_st)

        if hasattr(wx, 'dFz'):
            # with load transfer
            #Fz_front = self.Fz_front(wp_st) + wx.dFz
            #Fz_rear = self.Fz_rear(wp_st) - wx.dFz
            F_aero = wp_st.Cl0 * torch.sign(wx.v_x) +\
                wp_st.Cl1 * wx.v_x +\
                wp_st.Cl2 * wx.v_x * wx.v_x
            Fz_front = self.Fz_front(wp_st) - wx.dFz + F_aero
            Fz_rear = self.Fz_rear(wp_st) + wx.dFz + F_aero
        else:
            Fz_front = self.Fz_front(wp_st)
            Fz_rear = self.Fz_rear(wp_st)

        front_input = torch.stack([slip_angle_front, slip_ratio_front], dim=-1)
        front_scalers = self.mlp_front(front_input)
        rear_input = torch.stack([slip_angle_rear, slip_ratio_rear], dim=-1)
        rear_scalers = self.mlp_rear(rear_input)
        forces = self._tire_forces(front_scalers, rear_scalers, Fz_front, Fz_rear)
        return forces

    def get_parameters(self):
        return {
            'tires_front': self.mlp_front.parameters(),
            'tires_rear': self.mlp_rear.parameters(),
        }

    def get_parameters_vector(self):
        return torch.cat([self.mlp_front.parameters(), self.mlp_rear.parameters()], dim=-1)
        
        
