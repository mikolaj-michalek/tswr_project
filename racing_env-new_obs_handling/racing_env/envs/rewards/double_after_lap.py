import torch


class DoubleAfterLapReward:
    def __init__(self, env, progress_coeff=1.0, off_track_coeff=-1.0):
        self.env = env
        self.progress_coeff = progress_coeff
        self.off_track_coeff = off_track_coeff

    def __call__(self):
        slightly_off_track = (
            self.off_track_coeff
            * torch.nn.functional.relu(-self.env.signed_dist_to_edge + 0.05) ** 2.0
        )

        laps_finished = (
            self.env.ep_progress / self.env.track_lengths
        ).long().clamp(min=0, max=10).float()
        progress_scaler = torch.pow(2.0, laps_finished)
        progress_reward = (
            self.progress_coeff * progress_scaler * self.env.progress / self.env.dt
        )

        reward = progress_reward
        slightly_off_track_reward = progress_reward + slightly_off_track

        reward = reward / 10  # TODO investigate impact of this scaling
        slightly_off_track_reward = slightly_off_track_reward / 10

        reward = torch.where(
            self.env.slightly_off_track, slightly_off_track_reward, reward
        )
        reward = torch.where(self.env.off_track, self.off_track_coeff, reward)

        return reward