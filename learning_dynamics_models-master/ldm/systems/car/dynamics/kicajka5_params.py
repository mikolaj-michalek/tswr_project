import torch
import collections

Kicajka5ParamsList = ['x_norm', 'y_norm', 'x_scale', 'y_scale', 'sr_offset', 'sa_offset',
                      'Fx_offset', 'Fy_offset', 'cps_x', 'cps_y']
Kicajka5Params = collections.namedtuple('Kicajka5Params', Kicajka5ParamsList,
                                        defaults=[] * len(Kicajka5ParamsList))


class Kicajka5Parameters(torch.nn.Module):
    def __init__(self, n_train, n_up) -> None:
        super(Kicajka5Parameters, self).__init__()
        self.n_train = n_train
        self.n_up = n_up
        self.log_x_norm = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_norm = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.log_x_scale = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_scale = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.sr_offset = torch.nn.Parameter(torch.tensor([0.0]))
        self.sa_offset = torch.nn.Parameter(torch.tensor([0.0]))

        self.Fx_offset = torch.nn.Parameter(torch.tensor([0.0]))
        self.Fy_offset = torch.nn.Parameter(torch.tensor([0.0]))

        self.log_dv_x = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())
        self.log_dv_y = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())

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

        return Kicajka5Params(
            x_norm=x_norm,
            y_norm=y_norm,
            x_scale=x_scale,
            y_scale=y_scale,
            sr_offset=self.sr_offset,
            sa_offset=self.sa_offset,
            Fx_offset=self.Fx_offset,
            Fy_offset=self.Fy_offset,
            cps_x=cps_x,
            cps_y=cps_y,
        )