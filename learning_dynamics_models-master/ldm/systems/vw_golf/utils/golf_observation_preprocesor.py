import torch


class GolfObservationPreprocessor(torch.nn.Module):
    def __init__(self, max_velocity=1.0,
                       max_yaw_rate=1.0,
                       max_omega_wheels=1.0,
                       max_delta=1.0):
        super(GolfObservationPreprocessor, self).__init__()
        
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self.max_omega_wheels = max_omega_wheels
        self.max_delta = max_delta

    def forward(self, M):
        M = torch.cat([M[..., :1] / self.max_velocity,
                         M[..., 1:2] / self.max_velocity,
                         M[..., 2:3] / self.max_yaw_rate,
                         M[..., 3:-3], # pass potential state extenstions as is
                         M[..., -3:-2] / self.max_omega_wheels,
                         M[..., -2:-1] / self.max_omega_wheels,
                         M[..., -1:] / self.max_delta], dim=-1)
        return M
