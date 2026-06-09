import torch
import collections

DummyKicajkaParamsList = ['x_norm', 'y_norm', 'x_scale', 'y_scale', 'cps_x', 'cps_y']
DummyKicajkaParams = collections.namedtuple('PacejkaParams', DummyKicajkaParamsList, defaults=[] * len(DummyKicajkaParamsList))


class DummyKicajkaParameters(torch.nn.Module):
    def __init__(self, n_train) -> None:
        super(DummyKicajkaParameters, self).__init__()
        self.n_train = n_train
        self.log_x_norm = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_norm = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.log_x_scale = torch.nn.Parameter(torch.tensor([1.0]).log())
        self.log_y_scale = torch.nn.Parameter(torch.tensor([1.0]).log())

        self.log_v_x = torch.nn.Parameter((torch.linspace(1e-5, 1., self.n_train)).log())
        self.log_v_y = torch.nn.Parameter((torch.linspace(1e-5, 1., self.n_train)).log())

    def forward(self):
        return DummyKicajkaParams(
            x_norm=self.log_x_norm.exp(),
            y_norm=self.log_y_norm.exp(),
            x_scale=self.log_x_scale.exp(),
            y_scale=self.log_y_scale.exp(),
            cps_x=self.log_v_x.exp(),
            cps_y=self.log_v_y.exp(),
        )