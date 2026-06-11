import torch

from racing_env.envs.initializers.interface import InitializerInterface
from racing_env.utils.constants import CAR


class ZeroInitializer(InitializerInterface):
    def __init__(self, env):
        super().__init__(env)
        self.eps = 1e-6
        self.batch_idx = torch.arange(self.env.num_envs, device=self.env.device)

    def _get_start_point(self, start_at_zero=False):
        dev = self.env.device
        if start_at_zero:
            start_idx = torch.zeros(self.env.num_envs, dtype=torch.int64, device=dev)
        else:
            # ML
            #start_idx = (torch.rand(self.env.num_envs, device=dev) * self.env.track_size).int()
            start_idx = torch.ones(self.env.num_envs, dtype=torch.int64, device=dev) * 8

        start_x = self.env.track_x[self.batch_idx, start_idx]
        start_y = self.env.track_y[self.batch_idx, start_idx]
        start_yaw = self.env.track_heading[self.batch_idx, start_idx]

        if not start_at_zero:
            dist_to_centerline = (2 * torch.rand(self.env.num_envs, device=dev) - 1.) * \
                (self.env.track_width[self.batch_idx, start_idx] - CAR.WIDTH) / 2.
            start_x = start_x + dist_to_centerline * torch.sin(start_yaw)
            start_y = start_y + dist_to_centerline * torch.cos(start_yaw)

        last_s = self.env.track_s[self.batch_idx, start_idx]

        return start_x, start_y, start_yaw, last_s, start_idx

    def _get_initial_velocities_and_controls(self, start_at_zero=False):
        vx = CAR.MIN_VELOCITY
        vy = 0.0
        yaw_rate = 0.0
        omega_wheels = vx + self.eps
        omega_wheels_ref = omega_wheels + self.eps
        delta = 0.0
        delta_ref = delta + self.eps
        omega_wheels_ref_dot = 0.0
        return vx, vy, yaw_rate, omega_wheels, omega_wheels_ref, delta, delta_ref, omega_wheels_ref_dot

    def initialize(self, mask=None, start_at_zero=False):
        if mask is None:
            mask = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        start_x, start_y, start_yaw, last_s, start_idx = self._get_start_point(start_at_zero)
        vx, vy, yaw_rate, omega_wheels, omega_wheels_ref, delta, delta_ref, omega_wheels_ref_dot = \
            self._get_initial_velocities_and_controls(start_at_zero)


        # set start pose only for envs that are off track
        self.env.state[:, 0] = torch.where(mask, start_x, self.env.state[:, 0])
        self.env.state[:, 1] = torch.where(mask, start_y, self.env.state[:, 1])
        self.env.state[:, 2] = torch.where(mask, start_yaw, self.env.state[:, 2])
        self.env.state[:, 3] = torch.where(mask, vx, self.env.state[:, 3])
        self.env.state[:, 4] = torch.where(mask, vy, self.env.state[:, 4])
        self.env.state[:, 5] = torch.where(mask, yaw_rate, self.env.state[:, 5])
        self.env.state[:, 6] = torch.where(mask, omega_wheels, self.env.state[:, 6])
        self.env.state[:, 7] = torch.where(mask, omega_wheels_ref, self.env.state[:, 7])
        self.env.state[:, 8] = torch.where(mask, delta, self.env.state[:, 8])
        start_friction = self.env.friction[self.batch_idx, start_idx]
        self.env.state[:, 9] = torch.where(mask, start_friction, self.env.state[:, 9])
        self.env.state[:, 10] = torch.where(mask, start_friction, self.env.state[:, 10])
        self.env.state[:, 11] = torch.where(mask, delta_ref, self.env.state[:, 11])
        self.env.state[:, 12] = torch.where(mask, omega_wheels_ref_dot, self.env.state[:, 12])

        self.env.last_s = torch.where(mask, last_s, self.env.last_s)
        self.env.closest_idx = torch.where(mask, start_idx, self.env.closest_idx)