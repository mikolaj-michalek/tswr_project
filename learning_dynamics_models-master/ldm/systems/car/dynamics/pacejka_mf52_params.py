import torch
import collections

# Pacejka Magic Formula 5.2 (Pacejka, "Tyre and Vehicle Dynamics", 2nd ed., 2006)
# Parameters for combined-slip model using G_xα / G_yκ weighting functions.
# Pure-slip convention: slip angle α in degrees (consistent with codebase).
# G-function convention: α_S in degrees, κ_S dimensionless.

PacejkaMF52ParamsList = [
    # Pure-slip longitudinal
    'Bx', 'Cx', 'Dx', 'Ex', 'Shx', 'Svx',
    # Pure-slip lateral  (α in degrees)
    'By', 'Cy', 'Dy', 'Ey', 'Shy', 'Svy',
    # Combined-slip longitudinal weighting G_xα  (α_S in degrees)
    'Bxa', 'Cxa', 'Exa', 'SHxa',
    # Combined-slip lateral weighting G_yκ
    'Byk', 'Cyk', 'Eyk', 'SHyk',
]
PacejkaMF52Params = collections.namedtuple(
    'PacejkaMF52Params', PacejkaMF52ParamsList,
    defaults=[] * len(PacejkaMF52ParamsList)
)


class PacejkaMF52Parameters(torch.nn.Module):
    def __init__(self, randomize_init=0.0) -> None:
        super().__init__()

        # --- Pure-slip longitudinal ---
        self.log_Bx  = torch.nn.Parameter(torch.tensor([30.0]).log())   # Stiffness factor
        self.log_Cx  = torch.nn.Parameter(torch.tensor([0.3]).log())    # Shape factor
        self.log_Dx  = torch.nn.Parameter(torch.tensor([1.0]).log())    # Peak value
        self.Ex      = torch.nn.Parameter(torch.tensor([0.9]))          # Curvature factor
        self.Shx     = torch.nn.Parameter(torch.tensor([0.0]))          # Horizontal shift
        self.Svx     = torch.nn.Parameter(torch.tensor([0.0]))          # Vertical shift

        # --- Pure-slip lateral (α in degrees) ---
        self.log_By  = torch.nn.Parameter(torch.tensor([0.35]).log())   # Stiffness factor
        self.log_Cy  = torch.nn.Parameter(torch.tensor([0.4]).log())    # Shape factor
        self.log_Dy  = torch.nn.Parameter(torch.tensor([1.0]).log())    # Peak value
        self.Ey      = torch.nn.Parameter(torch.tensor([0.9]))          # Curvature factor
        self.Shy     = torch.nn.Parameter(torch.tensor([0.0]))          # Horizontal shift
        self.Svy     = torch.nn.Parameter(torch.tensor([0.0]))          # Vertical shift

        # --- G_xα: longitudinal combined-slip weighting (α_S in degrees) ---
        # MF 5.2: full magic-formula structure with curvature factor E in G
        self.log_Bxa = torch.nn.Parameter(torch.tensor([0.1]).log())    # Stiffness (1/deg)
        self.log_Cxa = torch.nn.Parameter(torch.tensor([1.0]).log())    # Shape factor
        self.Exa     = torch.nn.Parameter(torch.tensor([-0.5]))         # Curvature factor
        self.SHxa    = torch.nn.Parameter(torch.tensor([0.0]))          # Horizontal shift (deg)

        # --- G_yκ: lateral combined-slip weighting ---
        # MF 5.2: full magic-formula structure with curvature factor E in G
        self.log_Byk = torch.nn.Parameter(torch.tensor([3.0]).log())    # Stiffness (1/-)
        self.log_Cyk = torch.nn.Parameter(torch.tensor([1.0]).log())    # Shape factor
        self.Eyk     = torch.nn.Parameter(torch.tensor([-0.5]))         # Curvature factor
        self.SHyk    = torch.nn.Parameter(torch.tensor([0.0]))          # Horizontal shift

        if randomize_init > 0.0:
            self.randomize_parameters(scale=randomize_init)

    def randomize_parameters(self, scale=0.5):
        with torch.no_grad():
            for attr in ['log_Bx', 'log_Cx', 'log_Dx',
                         'log_By', 'log_Cy', 'log_Dy',
                         'log_Bxa', 'log_Cxa',
                         'log_Byk', 'log_Cyk']:
                p = getattr(self, attr)
                p.data = (p.exp() * (1. + scale * (2. * torch.rand_like(p) - 1.))).log()
            for attr in ['Ex', 'Ey', 'Exa', 'Eyk']:
                p = getattr(self, attr)
                p.data = p * (1. + scale * (2. * torch.rand_like(p) - 1.))

    def forward(self):
        return PacejkaMF52Params(
            Bx=self.log_Bx.exp(),   Cx=self.log_Cx.exp(),   Dx=self.log_Dx.exp(),
            Ex=self.Ex,             Shx=self.Shx,            Svx=self.Svx,
            By=self.log_By.exp(),   Cy=self.log_Cy.exp(),   Dy=self.log_Dy.exp(),
            Ey=self.Ey,             Shy=self.Shy,            Svy=self.Svy,
            Bxa=self.log_Bxa.exp(), Cxa=self.log_Cxa.exp(), Exa=self.Exa,
            SHxa=self.SHxa,
            Byk=self.log_Byk.exp(), Cyk=self.log_Cyk.exp(), Eyk=self.Eyk,
            SHyk=self.SHyk,
        )

    def get_parameters_vector(self):
        return torch.cat([
            self.log_Bx.unsqueeze(0).exp(),   self.log_Cx.unsqueeze(0).exp(),
            self.log_Dx.unsqueeze(0).exp(),   self.Ex.unsqueeze(0),
            self.Shx.unsqueeze(0),            self.Svx.unsqueeze(0),
            self.log_By.unsqueeze(0).exp(),   self.log_Cy.unsqueeze(0).exp(),
            self.log_Dy.unsqueeze(0).exp(),   self.Ey.unsqueeze(0),
            self.Shy.unsqueeze(0),            self.Svy.unsqueeze(0),
            self.log_Bxa.unsqueeze(0).exp(),  self.log_Cxa.unsqueeze(0).exp(),
            self.Exa.unsqueeze(0),            self.SHxa.unsqueeze(0),
            self.log_Byk.unsqueeze(0).exp(),  self.log_Cyk.unsqueeze(0).exp(),
            self.Eyk.unsqueeze(0),            self.SHyk.unsqueeze(0),
        ])
