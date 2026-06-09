import torch
import collections

ActuatorsParamsList = ['tau_omega', 'tau_delta']
ActuatorsParams = collections.namedtuple('ActuatorsParams', ActuatorsParamsList,
                                           defaults=(None,) * len(ActuatorsParamsList))

class ActuatorsParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(ActuatorsParameters, self).__init__()
        self.param_count = len(ActuatorsParamsList)

        self.register_buffer('tau_omega', torch.tensor([0.1], dtype=torch.float32))
        self.register_buffer('tau_delta', torch.tensor([0.05], dtype=torch.float32))

    def forward(self) -> ActuatorsParams:       
        named_tuple = ActuatorsParams(
            tau_delta=self.tau_delta,
            tau_omega=self.tau_omega,
        )
        return named_tuple