import torch


class Drift360Reward:
    def __init__(
        self,
        env,
        progress_coeff=1.0,
        centerline_coeff=-0.2,
        heading_coeff=-0.3,
        speed_coeff=0.05,
        off_track_coeff=-10.0,
    ):
        self.env = env
        self.progress_coeff = progress_coeff
        self.centerline_coeff = centerline_coeff
        self.heading_coeff = heading_coeff
        self.speed_coeff = speed_coeff
        self.off_track_coeff = off_track_coeff

    def __call__(self):
        state = self.env.state

        v_x = state[:, 3]

        # 1. Najważniejsze: progres po torze
        progress_reward = self.progress_coeff * self.env.progress / self.env.dt

        # 2. Mała nagroda za jazdę do przodu
        speed_reward = self.speed_coeff * torch.clamp(v_x, 0.0, 10.0)

        # 3. Kara za bycie daleko od środka toru
        centerline_penalty = self.centerline_coeff * torch.abs(
            self.env.signed_dist_to_centerline
        )

        # 4. Kara za zły kierunek względem toru
        heading_penalty = self.heading_coeff * torch.abs(self.env.heading_diff)

        reward = (
            progress_reward
            + speed_reward
            + centerline_penalty
            + heading_penalty
        )

        # 5. Kara za lekkie wyjechanie poza tor
        slightly_off_track_penalty = (
            -2.0
            * torch.nn.functional.relu(-self.env.signed_dist_to_edge + 0.05) ** 2.0
        )

        reward = torch.where(
            self.env.slightly_off_track,
            reward + slightly_off_track_penalty,
            reward,
        )

        # 6. Mocna kara za całkowity wyjazd
        reward = torch.where(
            self.env.off_track,
            torch.ones_like(reward) * self.off_track_coeff,
            reward,
        )

        return reward / 10.0