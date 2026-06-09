import torch

from ldm.systems.car.dynamics.single_track_params import DefaultSingleTrackParameters, SingleTrackParams

class VWGolfSingleTrackParameters(DefaultSingleTrackParameters):
    def __init__(self) -> None:
        super(VWGolfSingleTrackParameters, self).__init__()
        # Learnable parameters stored in log-space to guarantee positivity.
        # Defined as Parameters so they shadow the parent buffers of the same name.
        self.log_m = torch.nn.Parameter(torch.tensor([1200.]).log())
        self.log_I_z = torch.nn.Parameter(torch.tensor([2000.]).log())
        self.log_lr = torch.nn.Parameter(torch.tensor([1.5]).log())
        self.log_Cd0 = torch.nn.Parameter(torch.tensor([0.01]).log())
        self.log_Cd1 = torch.nn.Parameter(torch.tensor([0.01]).log())
        self.log_Cd2 = torch.nn.Parameter(torch.tensor([0.01]).log())

        self.register_buffer('L', torch.tensor([2.5]))

    def forward(self) -> SingleTrackParams:
        return SingleTrackParams(
            m=self.log_m.exp(),
            g=self.g,
            I_z=self.log_I_z.exp(),
            L=self.L,
            lr=self.log_lr.exp(),
            lf=self.L - self.log_lr.exp(),  # lf = L - lr
            Cd0=self.log_Cd0.exp(),
            Cd2=self.log_Cd2.exp(),
            Cd1=self.log_Cd1.exp(),
            eps=self.eps
        )