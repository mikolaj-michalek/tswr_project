
import torch


class DummyVFPreprocessor(torch.nn.Module):
    def __init__(self):
        super(DummyVFPreprocessor, self).__init__()
        
    def forward(self, M):
        """
            M : observations (batch_size, time_steps, observarions_channels)
        """
        return M[..., :-1]
