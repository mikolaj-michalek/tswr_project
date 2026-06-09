import torch

from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.state_wrapper import StateWrapper


class NeuralTireModel(BaseTireModel):
    def __init__(self, n_in: int = 6, nn=32, input_type: str = "vinvariant", *args, **kwargs) -> None:
        super().__init__()
        assert input_type in ["state", "vinvariant"]
        self.input_type = input_type

        self.n_in = 4 if input_type == "vinvariant" else n_in
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, nn),
            torch.nn.ReLU(),
            torch.nn.Linear(nn, nn),
            torch.nn.ReLU(),
            torch.nn.Linear(nn, 4),
        )

    def _tire_forces(self, scalers, Fz_front, Fz_rear):
        forces = torch.stack([
            Fz_front * scalers[..., 0],
            Fz_rear * scalers[..., 1],
            Fz_front * scalers[..., 2],
            Fz_rear * scalers[..., 3],
        ], dim=-1)
        return forces

    def tire_forces(self, slip_ratio_front, slip_angle_front,
                          slip_ratio_rear, slip_angle_rear, wp_st):
        input = torch.stack([slip_angle_front, slip_ratio_front, slip_angle_rear, slip_ratio_rear], dim=-1)
        scalers = self.mlp(input)
        Fz_front = self.Fz_front(wp_st)
        Fz_rear = self.Fz_rear(wp_st)
        forces = self._tire_forces(scalers, Fz_front, Fz_rear)
        return forces.unbind(-1)

    
    def forward(self, x, wp_st):
        input = x
        wx = StateWrapper(x)
        if self.input_type == "vinvariant":
            slip_angle_front = self.slip_angle_front_func(wx, wp_st)
            slip_ratio_front = self.slip_ratio_front_func(wx, wp_st)
            slip_angle_rear = self.slip_angle_rear_func(wx, wp_st)
            slip_ratio_rear = self.slip_ratio_rear_func(wx, wp_st)
            input = torch.stack([slip_angle_front, slip_ratio_front, slip_angle_rear, slip_ratio_rear], dim=-1)

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

        scalers = self.mlp(input)
        forces = self._tire_forces(scalers, Fz_front, Fz_rear)
        return forces

    def get_parameters(self):
        return {
            'tires': self.mlp.parameters(),
        }

    def get_parameters_vector(self):
        return self.mlp.parameters()
        
        
