import torch
import collections

Kicajka3ParamsList = ['x_norm', 'y_norm', 'x_scale', 'y_scale', 'cps_x', 'cps_y']
Kicajka3Params = collections.namedtuple('PacejkaParams', Kicajka3ParamsList, defaults=[] * len(Kicajka3ParamsList))


class Kicajka3Parameters(torch.nn.Module):
    def __init__(self, n_train, n_up) -> None:
        super(Kicajka3Parameters, self).__init__()
        self.n_train = n_train
        self.n_up = n_up
        self.log_x_norm = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_norm = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.log_x_scale = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_scale = torch.nn.Parameter(torch.tensor([1.0]).log())

        #self.log_dv_x = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())
        #self.log_dv_y = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())
        self.log_dv_x = torch.nn.Parameter((1.0 * (torch.linspace(1.0, 0.1, self.n_train)) ** 2. / self.n_train).log())
        self.log_dv_y = torch.nn.Parameter((1.5 * (torch.linspace(1.0, 0.1, self.n_train)) ** 2. / self.n_train).log())

    def forward(self):
        x_norm = self.log_x_norm.exp()
        y_norm = self.log_y_norm.exp()
        x_scale = self.log_x_scale.exp()
        y_scale = self.log_y_scale.exp()
       
        cps_up_x = torch.cumsum(self.log_dv_x.exp()[:self.n_up], dim=0)
        cps_down_x = torch.cumsum(-self.log_dv_x.exp()[self.n_up:], dim=0)

        cps_x = torch.cat([torch.zeros_like(cps_up_x[..., :1]),
                         cps_up_x,
                         cps_down_x + cps_up_x[..., -1:],
                         cps_down_x[..., -1:] + cps_up_x[..., -1:]], dim=-1)

        cps_up_y = torch.cumsum(self.log_dv_y.exp()[:self.n_up], dim=0)
        cps_down_y = torch.cumsum(-self.log_dv_y.exp()[self.n_up:], dim=0)

        cps_y = torch.cat([torch.zeros_like(cps_up_y[..., :1]),
                         cps_up_y,
                         cps_down_y + cps_up_y[..., -1:],
                         cps_down_y[..., -1:] + cps_up_y[..., -1:]], dim=-1)

        return Kicajka3Params(
            x_norm=x_norm,
            y_norm=y_norm,
            x_scale=x_scale,
            y_scale=y_scale,
            cps_x=cps_x,
            cps_y=cps_y,
        )

    def get_parameters_dict(self):
        return {
            'x_norm': self.log_x_norm.exp().item(),
            'y_norm': self.log_y_norm.exp().item(),
            'x_scale': self.log_x_scale.exp().item(),
            'y_scale': self.log_y_scale.exp().item(),
            'cps_x': self.forward().cps_x.detach().cpu().numpy(),
            'cps_y': self.forward().cps_y.detach().cpu().numpy(),
        }