import torch
import collections

TMeasyParamsList = [
    'df_x0', 's_mx', 'mu_mx', 's_gx', 'mu_gx',  # Longitudinal
    'df_y0', 's_my', 'mu_my', 's_gy', 'mu_gy'   # Lateral
]
TMeasyParams = collections.namedtuple('TMeasyParams', TMeasyParamsList, defaults=[] * len(TMeasyParamsList))

class TMeasyParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(TMeasyParameters, self).__init__()
        # Longitudinal Parameters
        self.log_df_x0 = torch.nn.Parameter(torch.tensor([10.0]).log())   # Initial stiffness (dFx0/Fz)
        self.log_s_mx = torch.nn.Parameter(torch.tensor([0.2]).log())    # Slip at max force
        self.log_mu_mx = torch.nn.Parameter(torch.tensor([1.]).log())   # Peak friction coefficient (FxM/Fz)
        self.log_s_gx = torch.nn.Parameter(torch.tensor([0.6]).log())    # Slip at full sliding
        self.log_mu_gx = torch.nn.Parameter(torch.tensor([0.9]).log())   # Sliding friction coefficient (FxG/Fz)

        # Lateral Parameters
        self.log_df_y0 = torch.nn.Parameter(torch.tensor([10.0]).log())   # Initial stiffness (dFy0/Fz)
        self.log_s_my = torch.nn.Parameter(torch.tensor([0.2]).log())    # Slip at max force
        self.log_mu_my = torch.nn.Parameter(torch.tensor([1.]).log())   # Peak friction coefficient (FyM/Fz)
        self.log_s_gy = torch.nn.Parameter(torch.tensor([0.6]).log())    # Slip at full sliding
        self.log_mu_gy = torch.nn.Parameter(torch.tensor([0.9]).log())   # Sliding friction coefficient (FyG/Fz)

    def forward(self):
        return TMeasyParams(
            df_x0=self.log_df_x0.exp(),
            s_mx=self.log_s_mx.exp(),
            mu_mx=self.log_mu_mx.exp(),
            s_gx=self.log_s_gx.exp(),
            mu_gx=self.log_mu_gx.exp(),
            df_y0=self.log_df_y0.exp(),
            s_my=self.log_s_my.exp(),
            mu_my=self.log_mu_my.exp(),
            s_gy=self.log_s_gy.exp(),
            mu_gy=self.log_mu_gy.exp()
        )

    def get_parameters_dict(self):
        return {
            'df_x0': self.log_df_x0.exp().item(),
            's_mx': self.log_s_mx.exp().item(),
            'mu_mx': self.log_mu_mx.exp().item(),
            's_gx': self.log_s_gx.exp().item(),
            'mu_gx': self.log_mu_gx.exp().item(),
            'df_y0': self.log_df_y0.exp().item(),
            's_my': self.log_s_my.exp().item(),
            'mu_my': self.log_mu_my.exp().item(),
            's_gy': self.log_s_gy.exp().item(),
            'mu_gy': self.log_mu_gy.exp().item()
        }

    def get_parameters_vector(self):
        return torch.cat([
            self.log_df_x0.unsqueeze(0).exp(),
            self.log_s_mx.unsqueeze(0).exp(),
            self.log_mu_mx.unsqueeze(0).exp(),
            self.log_s_gx.unsqueeze(0).exp(),
            self.log_mu_gx.unsqueeze(0).exp(),
            self.log_df_y0.unsqueeze(0).exp(),
            self.log_s_my.unsqueeze(0).exp(),
            self.log_mu_my.unsqueeze(0).exp(),
            self.log_s_gy.unsqueeze(0).exp(),
            self.log_mu_gy.unsqueeze(0).exp()
        ])