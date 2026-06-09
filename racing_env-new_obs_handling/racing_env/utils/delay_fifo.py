import torch


class DelayBufferFIFO(torch.nn.Module):
    def __init__(self, delay_time_in_samples, num_envs, device: torch.device = torch.device("cpu")):
        super(DelayBufferFIFO, self).__init__()
        self.delay_time_in_samples = delay_time_in_samples
        self.buffor = torch.zeros(num_envs, delay_time_in_samples, device=device)
        self.i = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def forward(self, x: torch.tensor):
        ans = self.buffor.gather(1, self.i.unsqueeze(1)).squeeze(1).clone()
        batch_idx = torch.arange(self.buffor.shape[0], device=self.buffor.device)
        self.buffor[batch_idx, self.i] = x.clone()
        self.i = (self.i + 1) % (self.delay_time_in_samples)

        return ans

    def reset(self, mask: torch.Tensor) -> None:
        """Clear buffer and index for envs where mask=True."""
        self.buffor = torch.where(mask[:, None], torch.zeros_like(self.buffor), self.buffor)
        self.i = torch.where(mask, torch.zeros_like(self.i), self.i)