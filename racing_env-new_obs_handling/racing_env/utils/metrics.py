import torch


class Metrics:
    """Pure rollout-wide accumulator. Call step() every step, pop_statistics() at end of rollout."""

    def __init__(self, num_envs: int = 1, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def step(self,
             reward: torch.Tensor,
             progress: torch.Tensor,
             off_track: torch.Tensor,
             slightly_off_track: torch.Tensor,
             dt: float) -> None:
        self._total_reward += reward.sum()
        self._total_progress += progress.sum()
        self._total_off_track += off_track.sum()
        self._total_slightly_off_track += slightly_off_track.sum()
        self._total_duration += dt * self.num_envs
        self._total_steps += self.num_envs

    def count_resets(self, n: int) -> None:
        self._n_resets += n

    def pop_statistics(self) -> dict | None:
        if self._total_steps.item() == 0:
            return None

        eps = 1e-8
        s = self._total_steps.item() + eps
        stats = {
            "avg_reward": self._total_reward.item() / s,
            "avg_progress": self._total_progress.item() / s,
            "avg_speed": self._total_progress.item() / (self._total_duration.item() + eps),
            "off_track_rate": self._total_off_track.item() / s,
            "slightly_off_track_rate": self._total_slightly_off_track.item() / s,
            "total_reward": self._total_reward.item(),
            "total_progress": self._total_progress.item(),
            "total_steps": self._total_steps.item(),
            "n_off_track": self._n_resets,
        }

        self.reset()
        return stats

    def reset(self) -> None:
        self._total_reward = torch.tensor(0.0, device=self.device)
        self._total_progress = torch.tensor(0.0, device=self.device)
        self._total_off_track = torch.tensor(0.0, device=self.device)
        self._total_slightly_off_track = torch.tensor(0.0, device=self.device)
        self._total_duration = torch.tensor(0.0, device=self.device)
        self._total_steps = torch.tensor(0, device=self.device)
        self._n_resets = 0
