import torch
import collections

# Parameters for the Dugoff model
DugoffParamsList = ['C_alpha', 'C_x', 'mu']
DugoffParams = collections.namedtuple('DugoffParams', DugoffParamsList, defaults=[] * len(DugoffParamsList))

class DugoffParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(DugoffParameters, self).__init__()
        # C_alpha: Cornering stiffness
        self.log_C_alpha = torch.nn.Parameter(torch.tensor([5.0]).log()) 
        # C_x: Longitudinal stiffness
        self.log_C_x = torch.nn.Parameter(torch.tensor([5.0]).log())
        # mu: Friction coefficient
        self.log_mu = torch.nn.Parameter(torch.tensor([1.0]).log())

    def forward(self):
        return DugoffParams(
            C_alpha=self.log_C_alpha.exp(),
            C_x=self.log_C_x.exp(),
            mu=self.log_mu.exp()
        )

    def get_parameters_dict(self):
        return {
            'C_alpha': self.log_C_alpha.exp().item(),
            'C_x':     self.log_C_x.exp().item(),
            'mu':      self.log_mu.exp().item(),
        }