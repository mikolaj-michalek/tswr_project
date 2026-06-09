import torch
import collections

# Pacejka Magic Formula 6.1 (Pacejka, "Tyre and Vehicle Dynamics", 3rd ed., 2012)
# Key structural differences vs MF 5.2:
#   1. G weighting functions use simplified cos-atan form WITHOUT the curvature
#      factor E (E is absorbed into the pure-slip formula only).
#   2. An additional lateral force term SV_yκ captures combined-slip lateral
#      load transfer; its amplitude DVyk is learnable.
# Pure-slip convention: slip angle α in degrees (consistent with codebase).

PacejkaMF61ParamsList = [
    # Pure-slip longitudinal
    'Bx', 'Cx', 'Dx', 'Ex', 'Shx', 'Svx',
    # Pure-slip lateral  (α in degrees)
    'By', 'Cy', 'Dy', 'Ey', 'Shy', 'Svy',
    # Combined-slip longitudinal weighting G_xα  (α_S in degrees, no E)
    'Bxa', 'Cxa', 'SHxa',
    # Combined-slip lateral weighting G_yκ  (no E)
    'Byk', 'Cyk', 'SHyk',
    # SVyκ amplitude (lateral force due to combined slip, zero without camber
    # in the theoretical model but learnable here)
    'DVyk',
]
PacejkaMF61Params = collections.namedtuple(
    'PacejkaMF61Params', PacejkaMF61ParamsList,
    defaults=[] * len(PacejkaMF61ParamsList)
)


class PacejkaMF61Parameters(torch.nn.Module):
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
        # MF 6.1: simplified cos(C·atan(B·α_S)) form, no E term in G
        self.log_Bxa = torch.nn.Parameter(torch.tensor([0.1]).log())    # Stiffness (1/deg)
        self.log_Cxa = torch.nn.Parameter(torch.tensor([1.0]).log())    # Shape factor
        self.SHxa    = torch.nn.Parameter(torch.tensor([0.0]))          # Horizontal shift (deg)

        # --- G_yκ: lateral combined-slip weighting ---
        # MF 6.1: simplified cos(C·atan(B·κ_S)) form, no E term in G
        self.log_Byk = torch.nn.Parameter(torch.tensor([3.0]).log())    # Stiffness (1/-)
        self.log_Cyk = torch.nn.Parameter(torch.tensor([1.0]).log())    # Shape factor
        self.SHyk    = torch.nn.Parameter(torch.tensor([0.0]))          # Horizontal shift

        # --- SVyκ amplitude ---
        # Lateral force added by combined slip (MF 6.1, Eq. 4.E80).
        # Theoretically zero for zero camber; kept learnable.
        self.DVyk    = torch.nn.Parameter(torch.tensor([0.0]))

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
            for attr in ['Ex', 'Ey']:
                p = getattr(self, attr)
                p.data = p * (1. + scale * (2. * torch.rand_like(p) - 1.))

    def forward(self):
        return PacejkaMF61Params(
            Bx=self.log_Bx.exp(),   Cx=self.log_Cx.exp(),   Dx=self.log_Dx.exp(),
            Ex=self.Ex,             Shx=self.Shx,            Svx=self.Svx,
            By=self.log_By.exp(),   Cy=self.log_Cy.exp(),   Dy=self.log_Dy.exp(),
            Ey=self.Ey,             Shy=self.Shy,            Svy=self.Svy,
            Bxa=self.log_Bxa.exp(), Cxa=self.log_Cxa.exp(), SHxa=self.SHxa,
            Byk=self.log_Byk.exp(), Cyk=self.log_Cyk.exp(), SHyk=self.SHyk,
            DVyk=self.DVyk,
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
            self.SHxa.unsqueeze(0),
            self.log_Byk.unsqueeze(0).exp(),  self.log_Cyk.unsqueeze(0).exp(),
            self.SHyk.unsqueeze(0),
            self.DVyk.unsqueeze(0),
        ])
