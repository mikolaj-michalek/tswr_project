import torch
import collections

# Define the Pacejka tire model parameters
PacejkaOffsetParamsList = ['Sx_p', 'Alpha_p', 'By', 'Cy', 'Dy', 'Ey', 'Bx', 'Cx', 'Dx', 'Ex', 'Svy', 'Svx', 'Shy', 'Shx']
PacejkaOffsetParams = collections.namedtuple('PacejkaOffsetParams', PacejkaOffsetParamsList, defaults=[] * len(PacejkaOffsetParamsList))

class PacejkaOffsetParameters(torch.nn.Module):
    def __init__(self, randomize_init=0.0) -> None:
        super(PacejkaOffsetParameters, self).__init__()
        self.log_Sx_p = torch.nn.Parameter(torch.tensor([0.05]).log())  # Peak slip ratio
        self.log_Alpha_p = torch.nn.Parameter(torch.tensor([2.0]).log())  # Peak slip angle (degrees)
        self.log_By = torch.nn.Parameter(torch.tensor([0.35]).log())  # Stiffness factor for lateral force
        self.log_Cy = torch.nn.Parameter(torch.tensor([0.4]).log())  # Shape factor for lateral force
        self.log_Dy = torch.nn.Parameter(torch.tensor([1.0]).log())  # Peak value for lateral force
        self.Ey = torch.nn.Parameter(torch.tensor([0.9]))  # Curvature factor for lateral force
        self.log_Bx = torch.nn.Parameter(torch.tensor([30.0]).log())  # Stiffness factor for longitudinal force
        self.log_Cx = torch.nn.Parameter(torch.tensor([0.3]).log())  # Shape factor for longitudinal force
        self.log_Dx = torch.nn.Parameter(torch.tensor([1.0]).log())  # Peak value for longitudinal force
        self.Ex = torch.nn.Parameter(torch.tensor([0.9]))  # Curvature factor for longitudinal force
        # # offsets
        self.Svy = torch.nn.Parameter(torch.tensor([0.0]))  # Lateral force offset
        self.Svx = torch.nn.Parameter(torch.tensor([0.0]))  # Longitudinal force offset
        self.Shy = torch.nn.Parameter(torch.tensor([0.0]))  # Lateral force offset
        self.Shx = torch.nn.Parameter(torch.tensor([0.0]))  # Longitudinal force offset

        if randomize_init > 0.0:
            self.randomize_parameters(scale=randomize_init)

    def randomize_parameters(self, scale=0.5):
        with torch.no_grad():
            self.log_Sx_p.data = (self.log_Sx_p.exp() * (1. + scale * (2. * torch.rand_like(self.log_Sx_p) - 1.))).log()
            self.log_Alpha_p.data = (self.log_Alpha_p.exp() * (1. + scale * (2. * torch.rand_like(self.log_Alpha_p) - 1.))).log()
            self.log_By.data = (self.log_By.exp() * (1. + scale * (2. * torch.rand_like(self.log_By) - 1.))).log()
            self.log_Cy.data = (self.log_Cy.exp() * (1. + scale * (2. * torch.rand_like(self.log_Cy) - 1.))).log()
            self.log_Dy.data = (self.log_Dy.exp() * (1. + scale * (2. * torch.rand_like(self.log_Dy) - 1.))).log()
            self.Ey.data = self.Ey * (1. + scale * (2. * torch.rand_like(self.Ey) - 1.))
            self.log_Bx.data = (self.log_Bx.exp() * (1. + scale * (2. * torch.rand_like(self.log_Bx) - 1.))).log()
            self.log_Cx.data = (self.log_Cx.exp() * (1. + scale * (2. * torch.rand_like(self.log_Cx) - 1.))).log()
            self.log_Dx.data = (self.log_Dx.exp() * (1. + scale * (2. * torch.rand_like(self.log_Dx) - 1.))).log()
            self.Ex.data = self.Ex * (1. + scale * (2. * torch.rand_like(self.Ex) - 1.))

    def forward(self):
        return PacejkaOffsetParams(
            Sx_p=self.log_Sx_p.exp(),
            Alpha_p=self.log_Alpha_p.exp(),
            By=self.log_By.exp(),
            Cy=self.log_Cy.exp(),
            Dy=self.log_Dy.exp(),
            Ey=self.Ey,
            Bx=self.log_Bx.exp(),
            Cx=self.log_Cx.exp(),
            Dx=self.log_Dx.exp(),
            Ex=self.Ex,
            # # offsets
            Svy=self.Svy,
            Svx=self.Svx,
            Shy=self.Shy,
            Shx=self.Shx
        )

    def get_parameters_vector(self):
        return torch.cat([
            self.log_Sx_p.unsqueeze(0).exp(),
            self.log_Alpha_p.unsqueeze(0).exp(),
            self.log_By.unsqueeze(0).exp(),
            self.log_Cy.unsqueeze(0).exp(),
            self.log_Dy.unsqueeze(0).exp(),
            self.Ey.unsqueeze(0),
            self.log_Bx.unsqueeze(0).exp(),
            self.log_Cx.unsqueeze(0).exp(),
            self.log_Dx.unsqueeze(0).exp(),
            self.Ex.unsqueeze(0),
            # # offsets
            self.Svy.unsqueeze(0),
            self.Svx.unsqueeze(0),
            self.Shy.unsqueeze(0),
            self.Shx.unsqueeze(0),
        ])