import torch
import collections

# Linear tire model parameters
LinearParamsList = ['C_alpha', 'C_x', 'mu']
LinearParams = collections.namedtuple('LinearParams', LinearParamsList, defaults=[] * len(LinearParamsList))

class LinearParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(LinearParameters, self).__init__()
        # C_alpha: Cornering stiffness (N/rad normalized by Fz)
        self.log_C_alpha = torch.nn.Parameter(torch.tensor([5.0]).log()) 
        # C_x: Longitudinal stiffness (normalized by Fz)
        self.log_C_x = torch.nn.Parameter(torch.tensor([5.0]).log())
        # mu: Maximum friction coefficient
        self.log_mu = torch.nn.Parameter(torch.tensor([1.0]).log())

    def forward(self):
        return LinearParams(
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

    def get_parameters_vector(self):
        return torch.cat([
            self.log_C_alpha.unsqueeze(0).exp(),
            self.log_C_x.unsqueeze(0).exp(),
            self.log_mu.unsqueeze(0).exp()
        ])