import torch
import collections

SingleTrackParamsList = ['m', 'g', 'I_z', 'L', 'lr', 'lf', 'Cd0', 'Cd2', 'Cd1',
                         'I_e', 'K_fi', 'b1', 'b0', 'R', 'eps']
SingleTrackParams = collections.namedtuple('SingleTrackParams', SingleTrackParamsList,
                                           defaults=(None,) * len(SingleTrackParamsList))

class DefaultSingleTrackParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(DefaultSingleTrackParameters, self).__init__()
        self.register_buffer('m', torch.tensor([5.1]))
        self.register_buffer('g', torch.tensor([9.81]))
        self.register_buffer('L', torch.tensor([0.33]))
        self.register_buffer('eps', torch.tensor([1e-6]))

        self.register_buffer('I_z', torch.tensor([0.1]))
        self.register_buffer('lr', torch.tensor([0.145]))

        # assume that the drag is negligible
        self.register_buffer('Cd0', torch.tensor([1e-9]))
        self.register_buffer('Cd1', torch.tensor([1e-9]))
        self.register_buffer('Cd2', torch.tensor([1e-9]))

        self.register_buffer('I_e', torch.tensor([0.2]))
        self.register_buffer('K_fi', torch.tensor([0.90064745]))
        self.register_buffer('b0', torch.tensor([0.50421894]))
        self.register_buffer('b1', torch.tensor([0.304115174]))
        self.register_buffer('R', torch.tensor([0.05]))

        

    def forward(self) -> SingleTrackParams:
        return SingleTrackParams(
            m=self.m,
            g=self.g,
            I_z=self.I_z,
            L=self.L,
            lr=self.lr,
            lf=self.L - self.lr,  # lf = L - lr
            Cd0=self.Cd0,
            Cd2=self.Cd2,
            Cd1=self.Cd1,
            I_e=self.I_e,
            K_fi=self.K_fi,
            b1=self.b1,
            b0=self.b0,
            R=self.R,
            eps=self.eps
        )

    def get_parameters_dict(self):
        return {
            'm': self.m.item(),
            'g': self.g.item(),
            'I_z': self.I_z.item(),
            'L': self.L.item(),
            'lr': self.lr.item(),
            'lf': (self.L - self.lr).item(),
            'Cd0': self.Cd0.item(),
            'Cd2': self.Cd2.item(),
            'Cd1': self.Cd1.item(),
            'I_e': self.I_e.item(),
            'K_fi': self.K_fi.item(),
            'b1': self.b1.item(),
            'b0': self.b0.item(),
            'R': self.R.item(),
            'eps': self.eps.item()
        }

    def get_parameters_vector(self):
        return torch.cat([
            self.I_z.unsqueeze(0),
            self.lr.unsqueeze(0),
            self.Cd0.unsqueeze(0),
            self.Cd1.unsqueeze(0),
            self.Cd2.unsqueeze(0),
            self.I_e.unsqueeze(0),
            self.K_fi.unsqueeze(0),
            self.b1.unsqueeze(0),
            self.b0.unsqueeze(0),
            self.R.unsqueeze(0)
        ], dim=-1)