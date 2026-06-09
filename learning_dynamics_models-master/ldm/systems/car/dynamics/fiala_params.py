import torch
import collections

# Physical parameters for the Fiala model
FialaParamsList = ['C_alpha', 'C_x', 'mu']
FialaParams = collections.namedtuple('FialaParams', FialaParamsList, defaults=[] * len(FialaParamsList))

class FialaParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(FialaParameters, self).__init__()
        # C_alpha: Lateral cornering stiffness
        self.log_C_alpha = torch.nn.Parameter(torch.tensor([20.0]).log()) 
        # C_x: Longitudinal slip stiffness
        self.log_C_x = torch.nn.Parameter(torch.tensor([20.0]).log())
        # mu: Friction coefficient
        self.log_mu = torch.nn.Parameter(torch.tensor([1.0]).log())

    def forward(self):
        return FialaParams(
            C_alpha=self.log_C_alpha.exp(),
            C_x=self.log_C_x.exp(),
            mu=self.log_mu.exp()
        )

    def get_parameters_vector(self):
        return torch.cat([
            self.log_C_alpha.unsqueeze(0).exp(),
            self.log_C_x.unsqueeze(0).exp(),
            self.log_mu.unsqueeze(0).exp()
        ])