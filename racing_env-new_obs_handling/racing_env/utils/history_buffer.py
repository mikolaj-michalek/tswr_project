import torch

class HistoryBuffer(torch.nn.Module):
    def __init__(self, history_length, num_envs, obs_dim, device: torch.device = torch.device("cpu")):
        super(HistoryBuffer, self).__init__()
        self.history_length = history_length
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device

        # Buffer shape: [num_envs, history_length, obs_dim]
        self.buffer = torch.zeros(num_envs, history_length, obs_dim, device=device)
        self.i = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.last_input = torch.zeros(num_envs, obs_dim, device=device)

    def add(self, x: torch.Tensor):
        """
        x: [num_envs, obs_dim]
        """
        indices = torch.arange(self.num_envs, device=self.device)
        self.buffer[indices, self.i] = x
        self.last_input = x
        self.i = (self.i + 1) % self.history_length

    def get_history(self):
        """
        Returns: [num_envs, history_length, obs_dim] with newest at index 0.
        """
        # Double buffer along history dim
        double_buffer = torch.cat([self.buffer, self.buffer], dim=1)
        indices = self.i.unsqueeze(1) + torch.arange(self.history_length, device=self.device)
        env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        history = double_buffer[env_indices, indices]  # [num_envs, history_length, obs_dim]
        history = history.flip(1)  # reverse to have newest first
        return history.clone()

    def reset(self, env_reset):
        """
        env_reset: [num_envs] bool tensor
        """
        self.buffer = torch.where(env_reset[:, None, None], torch.zeros_like(self.buffer), self.buffer)
        self.last_input = torch.where(env_reset[:, None], torch.zeros_like(self.last_input), self.last_input)
        self.i = torch.where(env_reset, torch.zeros_like(self.i), self.i)

    def get_last_input(self):
        """
        Returns: [num_envs, obs_dim]
        """
        return self.last_input.clone()


if __name__ == "__main__":
    # Test the history buffer
    history_length = 3
    num_envs = 2

    history_buffer = HistoryBuffer(history_length, num_envs)

    for i in range(20):
        x = torch.tensor([[i], [i + 1]], dtype=torch.float32)
        history_buffer.add(x)
        if i % 5 == 0:
            env_reset = torch.tensor([0, 1], dtype=torch.bool)
            history_buffer.reset(env_reset)
        print(f"Step {i+1}:")
        print(history_buffer.get_history())
        print(history_buffer.get_last_input())
