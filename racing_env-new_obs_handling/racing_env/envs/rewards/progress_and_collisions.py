import torch


class ProgressAndCollisionReward:
    def __init__(self, env, progress_coeff=1.0, off_track_coeff=-10.0, sec_to_double_reward=5.0):
        self.env = env
        self.progress_coeff = progress_coeff
        self.off_track_coeff = off_track_coeff
        self.sec_to_double_reward = sec_to_double_reward


    def __call__(self):        
        off_track_reward = self.off_track_coeff * torch.nn.functional.relu(-self.env.signed_dist_to_edge + 0.05)**2.

        progress_multiplier = 1 + (self.env.steps_no_off_track * self.env.dt / self.sec_to_double_reward)
        progress_reward = self.progress_coeff * self.env.progress / self.env.dt * progress_multiplier

        reward = progress_reward + off_track_reward

        reward = reward / 10 # TODO investigate impact of this scaling

        reward = torch.where(self.env.slightly_off_track, off_track_reward, reward)
        reward = torch.where(self.env.off_track, off_track_reward, reward)

        return reward

    