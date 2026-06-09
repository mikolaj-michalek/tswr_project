import torch

from racing_env.envs.initializers.zero import ZeroInitializer
from racing_env.utils.constants import CAR


class RandomInitializer(ZeroInitializer):
    def __init__(self, env):
        super().__init__(env)

    def _get_start_point(self, start_at_zero=False):
        start_x, start_y, start_yaw, last_s, start_idx = super()._get_start_point(start_at_zero)
        if not start_at_zero:
            dev = self.env.device
            batch_idx = torch.arange(self.env.num_envs, device=dev)
            dist_to_centerline = (2 * torch.rand(self.env.num_envs, device=dev) - 1.) * \
                self.env.track_width[batch_idx, start_idx] / 2.
            start_x = start_x + dist_to_centerline * torch.sin(start_yaw)
            start_y = start_y + dist_to_centerline * torch.cos(start_yaw)

        return start_x, start_y, start_yaw, last_s, start_idx

    def _get_initial_velocities_and_controls(self, start_at_zero=False):
        if start_at_zero:
            return super()._get_initial_velocities_and_controls(start_at_zero)
        vx = torch.rand(self.env.num_envs, device=self.env.device) * (5.0 - CAR.MIN_VELOCITY) + CAR.MIN_VELOCITY
        vy = (torch.rand(self.env.num_envs, device=self.env.device) * 2.0 - 1.0) * vx * 0.5
        yaw_rate = (torch.rand(self.env.num_envs, device=self.env.device) * 2.0 - 1.0) * 1.0
        omega_wheels = vx + torch.rand(self.env.num_envs, device=self.env.device) * 0.5
        omega_wheels_ref = omega_wheels + torch.rand(self.env.num_envs, device=self.env.device) * 0.5
        delta = (torch.rand(self.env.num_envs, device=self.env.device) * 2.0 - 1.0) * CAR.MAX_STEERING
        delta_ref = delta + (torch.rand(self.env.num_envs, device=self.env.device) * 2.0 - 1.0) * 0.1
        delta_ref = torch.clamp(delta_ref, -CAR.MAX_STEERING, CAR.MAX_STEERING)
        omega_wheels_ref_dot = (2. * torch.rand(self.env.num_envs, device=self.env.device) - 1.) * CAR.MAX_OMEGA_DOT
        return vx, vy, yaw_rate, omega_wheels, omega_wheels_ref, delta, delta_ref, omega_wheels_ref_dot