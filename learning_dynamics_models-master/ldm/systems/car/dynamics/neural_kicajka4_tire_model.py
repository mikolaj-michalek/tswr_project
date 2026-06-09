import torch

from ldm.systems.car.dynamics.kicajka4_params import Kicajka4Parameters, Kicajka4Params
from ldm.systems.car.dynamics.kicajka4_tire_model import Kicajka4TireModel
from ldm.systems.car.dynamics.state_wrapper import StateWrapper

class NeuralKicajka4TireModel(Kicajka4TireModel):
    def __init__(self, n_in=6, n=10, n_up=7, nn=16, input_type="vinvariant", *args, **kwargs) -> None:
        assert input_type in ["state", "vinvariant"]
        super().__init__(n, n_up)
        self.input_type = input_type
        self.n_in = 4 if input_type == "vinvariant" else n_in
        self.kicajka_parameters_nn = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, nn),
            torch.nn.ReLU(),
            torch.nn.Linear(nn, nn),
            torch.nn.ReLU(),
            torch.nn.Linear(nn, (2*n + 6) * 2, bias=False),
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
        scalers = self.kicajka_parameters_nn(input)
        Fz_front = self.Fz_front(wp_st)
        Fz_rear = self.Fz_rear(wp_st)
        forces = self._tire_forces(scalers, Fz_front, Fz_rear)
        return forces.unbind(-1)

    def forward(self, x, wp_st):
        wx = StateWrapper(x)
        slip_angle_front = self.slip_angle_front_func(wx, wp_st)
        slip_ratio_front = self.slip_ratio_front_func(wx, wp_st)
        slip_angle_rear = self.slip_angle_rear_func(wx, wp_st)
        slip_ratio_rear = self.slip_ratio_rear_func(wx, wp_st)

        input = x
        if self.input_type == "vinvariant":
            input = torch.stack([slip_angle_front, slip_ratio_front, slip_angle_rear, slip_ratio_rear], dim=-1)
        kicajka_params = self.kicajka_parameters_nn(input)

        front_tire_params = self.front_tire_model_parameters(kicajka_params[..., :kicajka_params.shape[-1]//2])
        rear_tire_params = self.rear_tire_model_parameters(kicajka_params[..., kicajka_params.shape[-1]//2:])

        fric_xf, fric_yf = self.tire_forces_model(slip_angle_front, slip_ratio_front,
                                                  front_tire_params)
        fric_xr, fric_yr = self.tire_forces_model(slip_angle_rear, slip_ratio_rear,
                                                  rear_tire_params)
        Fxr = self.Fz_rear(wp_st) * fric_xr
        Fyr = self.Fz_rear(wp_st) * fric_yr
        Fxf = self.Fz_front(wp_st) * fric_xf
        Fyf = self.Fz_front(wp_st) * fric_yf
        tire_forces = Fyf, Fyr, Fxf, Fxr
        return torch.stack(tire_forces, dim=-1)
