import torch


class ProgressCollisionReward:
    def __init__(self, env, progress_coeff=1.0):
        self.env = env
        self.progress_coeff = progress_coeff


    def __call__(self):        
        progress_reward = self.progress_coeff * self.env.progress / self.env.dt
        reward = progress_reward / 3.
        reward = torch.where(self.env.off_track, -torch.ones_like(reward), reward)
        return reward

    