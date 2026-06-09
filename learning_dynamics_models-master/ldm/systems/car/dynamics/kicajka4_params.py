import torch
import collections

Kicajka4ParamsList = ['x_norm', 'y_norm', 'x_scale', 'y_scale', 'sr_offset', 'sa_offset', 'cps_x', 'cps_y']
Kicajka4Params = collections.namedtuple('Kicajka4Params', Kicajka4ParamsList, defaults=[] * len(Kicajka4ParamsList))


class Kicajka4Parameters(torch.nn.Module):
    def __init__(self, n_train, n_up, randomize_init=0.) -> None:
        super(Kicajka4Parameters, self).__init__()
        self.n_train = n_train
        self.n_up = n_up
        self.log_x_norm = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_norm = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.log_x_scale = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_scale = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.sr_offset = torch.nn.Parameter(torch.tensor([0.0]))
        self.sa_offset = torch.nn.Parameter(torch.tensor([0.0]))

        #self.log_dv_x = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())
        #self.log_dv_y = torch.nn.Parameter((1.5 * torch.ones(self.n_train) / self.n_train).log())
        self.log_dv_x = torch.nn.Parameter((1.0 * (torch.linspace(1.0, 0.1, self.n_train)) ** 2. / self.n_train).log())
        self.log_dv_y = torch.nn.Parameter((1.5 * (torch.linspace(1.0, 0.1, self.n_train)) ** 2. / self.n_train).log())
        if randomize_init:
            self.randomize_parameters(scale=randomize_init)

    def randomize_parameters(self, scale=0.5):
        with torch.no_grad():
            self.log_x_norm.data = (self.log_x_norm.exp() * (1. + scale * (2. * torch.rand_like(self.log_x_norm) - 1.))).log()
            self.log_y_norm.data = (self.log_y_norm.exp() * (1. + scale * (2. * torch.rand_like(self.log_y_norm) - 1.))).log()
            self.log_x_scale.data = (self.log_x_scale.exp() * (1. + scale * (2. * torch.rand_like(self.log_x_scale) - 1.))).log()
            self.log_y_scale.data = (self.log_y_scale.exp() * (1. + scale * (2. * torch.rand_like(self.log_y_scale) - 1.))).log()   
            self.log_dv_x.data = (self.log_dv_x.exp() * (1. + scale * (2. * torch.rand_like(self.log_dv_x) - 1.))).log()
            self.log_dv_y.data = (self.log_dv_y.exp() * (1. + scale * (2. * torch.rand_like(self.log_dv_y) - 1.))).log()

    def forward(self, dparams=None):
        # Apply dparams offsets if provided, otherwise use zeros
        if dparams is not None:
            log_x_norm = self.log_x_norm + dparams[..., 0]
            log_y_norm = self.log_y_norm + dparams[..., 1]
            log_x_scale = self.log_x_scale + dparams[..., 2]
            log_y_scale = self.log_y_scale + dparams[..., 3]
            sr_offset = self.sr_offset + dparams[..., 4]
            sa_offset = self.sa_offset + dparams[..., 5]
            log_dv_x = self.log_dv_x + dparams[..., 6:6 + self.n_train]
            log_dv_y = self.log_dv_y + dparams[..., 6 + self.n_train:]
        else:
            log_x_norm, log_y_norm = self.log_x_norm, self.log_y_norm
            log_x_scale, log_y_scale = self.log_x_scale, self.log_y_scale
            sr_offset, sa_offset = self.sr_offset, self.sa_offset
            log_dv_x, log_dv_y = self.log_dv_x, self.log_dv_y

        # Exponentiate log parameters
        x_norm, y_norm = log_x_norm.exp(), log_y_norm.exp()
        x_scale, y_scale = log_x_scale.exp(), log_y_scale.exp()

        # Helper function for computing control points
        def compute_cps(log_dv):
            dv = log_dv.exp()
            cps_up = torch.cumsum(dv[..., :self.n_up], dim=0)
            cps_down = torch.cumsum(-dv[..., self.n_up:], dim=0)
            return torch.cat([
                torch.zeros_like(cps_up[..., :1]),
                cps_up,
                cps_down + cps_up[..., -1:],
                cps_down[..., -1:] + cps_up[..., -1:]
            ], dim=-1)

        return Kicajka4Params(
            x_norm=x_norm,
            y_norm=y_norm,
            x_scale=x_scale,
            y_scale=y_scale,
            sr_offset=sr_offset,
            sa_offset=sa_offset,
            cps_x=compute_cps(log_dv_x),
            cps_y=compute_cps(log_dv_y),
        )

    def get_parameters_dict(self):
        return {
            'x_norm': self.log_x_norm.exp().item(),
            'y_norm': self.log_y_norm.exp().item(),
            'x_scale': self.log_x_scale.exp().item(),
            'y_scale': self.log_y_scale.exp().item(),
            'sr_offset': self.sr_offset.item(),
            'sa_offset': self.sa_offset.item(),
            'cps_x': self.forward().cps_x.detach().cpu().numpy(),
            'cps_y': self.forward().cps_y.detach().cpu().numpy(),
        }