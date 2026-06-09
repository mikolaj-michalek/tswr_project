import torch
import collections

KicajkaParamsList = ['x_norm', 'y_norm', 'cps']
KicajkaParams = collections.namedtuple('PacejkaParams', KicajkaParamsList, defaults=[] * len(KicajkaParamsList))


class KicajkaParameters(torch.nn.Module):
    def __init__(self, n_train, n_up) -> None:
        super(KicajkaParameters, self).__init__()
        self.n_train = n_train
        self.n_up = n_up
        self.log_x_norm = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_norm = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.log_dv = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())

    def forward(self):
        x_norm = self.log_x_norm.exp()
        y_norm = self.log_y_norm.exp()
       
        cps_up = torch.cumsum(self.log_dv.exp()[:self.n_up], dim=0)
        cps_down = torch.cumsum(-self.log_dv.exp()[self.n_up:], dim=0)

        cps = torch.cat([torch.zeros_like(cps_up[..., :1]),
                         cps_up,
                         cps_down + cps_up[..., -1:],
                         cps_down[..., -1:] + cps_up[..., -1:]], dim=-1)

        return KicajkaParams(
            x_norm=x_norm,
            y_norm=y_norm,
            cps=cps,
        )