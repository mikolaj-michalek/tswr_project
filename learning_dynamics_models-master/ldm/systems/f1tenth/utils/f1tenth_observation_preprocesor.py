import torch

# ─────────────────────────────────────────────────────────────────────────────
# F1tenth dimension constants
#   STATE_DIM   : v_x, v_y, r, friction_front, friction_rear
#   CONTROL_DIM : omega_front, omega_rear, delta
#   N_DROPPED   : entries removed by CarObservationPreprocessor
#                 (duplicate rear friction at index 4, and rear omega at index -3)
#   OBS_DIM     : dimensionality of the preprocessor OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM   = 5
CONTROL_DIM = 3
N_DROPPED   = 2   # rear-friction duplicate + rear omega wheel
OBS_DIM     = STATE_DIM + CONTROL_DIM - N_DROPPED


class CarObservationPreprocessor(torch.nn.Module):
    def __init__(self, max_velocity=1.0,
                       max_yaw_rate=1.0,
                       max_omega_wheels=1.0,
                       max_delta=1.0):
        super(CarObservationPreprocessor, self).__init__()
        
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self.max_omega_wheels = max_omega_wheels
        self.max_delta = max_delta

    def forward(self, M):
        M = torch.cat([M[..., :1] / self.max_velocity,
                         M[..., 1:2] / self.max_velocity,
                         M[..., 2:3] / self.max_yaw_rate,
                         M[..., 3:4] - 0.8, # friction is around 0.8, so we center it around 0
                         M[..., 5:-3], # pass potential state extenstions as is
                         # ignore rear omega wheels because front and rear omega wheels are the same
                         M[..., -2:-1] / self.max_omega_wheels,
                         M[..., -1:] / self.max_delta], dim=-1)

        return M
