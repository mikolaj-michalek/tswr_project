import torch
import collections

# Define the Pacejka tire model parameters
DecoupledPacejkaParamsList = ['By', 'Cy', 'Dy', 'Ey', 'Bx', 'Cx', 'Dx', 'Ex']
DecoupledPacejkaParams = collections.namedtuple('DecoupledPacejkaParams', DecoupledPacejkaParamsList, defaults=[] * len(DecoupledPacejkaParamsList))

class DecoupledPacejkaParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(DecoupledPacejkaParameters, self).__init__()
        self.log_By = torch.nn.Parameter(torch.tensor([30.]).log())  # Stiffness factor for lateral force
        self.log_Cy = torch.nn.Parameter(torch.tensor([0.4]).log())  # Shape factor for lateral force
        self.log_Dy = torch.nn.Parameter(torch.tensor([1.0]).log())  # Peak value for lateral force
        self.Ey = torch.nn.Parameter(torch.tensor([0.9]))  # Curvature factor for lateral force
        self.log_Bx = torch.nn.Parameter(torch.tensor([30.0]).log())  # Stiffness factor for longitudinal force
        self.log_Cx = torch.nn.Parameter(torch.tensor([0.3]).log())  # Shape factor for longitudinal force
        self.log_Dx = torch.nn.Parameter(torch.tensor([1.0]).log())  # Peak value for longitudinal force
        self.Ex = torch.nn.Parameter(torch.tensor([0.9]))  # Curvature factor for longitudinal force

    def forward(self):
        return DecoupledPacejkaParams(
            By=self.log_By.exp(),
            Cy=self.log_Cy.exp(),
            Dy=self.log_Dy.exp(),
            Ey=self.Ey,
            Bx=self.log_Bx.exp(),
            Cx=self.log_Cx.exp(),
            Dx=self.log_Dx.exp(),
            Ex=self.Ex,
        )

    def get_parameters_vector(self):
        return torch.cat([
            self.log_By.unsqueeze(0).exp(),
            self.log_Cy.unsqueeze(0).exp(),
            self.log_Dy.unsqueeze(0).exp(),
            self.Ey.unsqueeze(0),
            self.log_Bx.unsqueeze(0).exp(),
            self.log_Cx.unsqueeze(0).exp(),
            self.log_Dx.unsqueeze(0).exp(),
            self.Ex.unsqueeze(0),
        ])