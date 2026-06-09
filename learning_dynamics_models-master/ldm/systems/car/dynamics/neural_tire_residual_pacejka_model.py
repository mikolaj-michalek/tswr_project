from ldm.systems.car.dynamics.base_tire_model import BaseTireModel
from ldm.systems.car.dynamics.pacejka_tire_model import PacejkaTireModel
from ldm.systems.car.dynamics.neural_tire_model import NeuralTireModel

class NeuralTireResidualPacejkaModel(BaseTireModel):
    def __init__(self, n_in: int = 4, input_type: str = "vinvariant", *args, **kwargs) -> None:
        super().__init__()
        self.neural_tire = NeuralTireModel(n_in=n_in, input_type=input_type)
        self.pacejka_tire = PacejkaTireModel()

    def tire_forces(self, slip_ratio_front, slip_angle_front,
               slip_ratio_rear, slip_angle_rear, wp_st):
        Fy_f_nn, Fy_r_nn, Fx_f_nn, Fx_r_nn = self.neural_tire.tire_forces(slip_ratio_front, slip_angle_front,
                                                                          slip_ratio_rear, slip_angle_rear, wp_st)
        Fy_f_pc, Fy_r_pc, Fx_f_pc, Fx_r_pc = self.pacejka_tire.tire_forces(slip_ratio_front, slip_angle_front,
                                                                           slip_ratio_rear, slip_angle_rear, wp_st)
        return Fy_f_pc + Fy_f_nn, Fy_r_pc + Fy_r_nn, Fx_f_pc + Fx_f_nn, Fx_r_pc + Fx_r_nn
        #return Fy_f_pc, Fy_r_pc, Fx_f_pc, Fx_r_pc
        #return Fy_f_nn, Fy_r_nn, Fx_f_nn, Fx_r_nn

    def forward(self, x, wp_st):
        forces = self.neural_tire(x, wp_st)
        forces += self.pacejka_tire(x, wp_st)
        return forces