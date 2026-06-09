import torch


class AcrobotObservationPreprocessor(torch.nn.Module):
    def __init__(self,
                 max_dq1: float = 1.0,
                 max_dq2: float = 1.0,
                 max_u: float = 1.0,
                 drop_control : bool = False,
                 only_control: bool = False,
                 drop_pos: bool = False,
                 compile: bool = False):
        super(AcrobotObservationPreprocessor, self).__init__()

        self.max_dq1 = max_dq1
        self.max_dq2 = max_dq2
        self.max_u = max_u
        self.drop_control = drop_control
        self.only_control = only_control
        self.drop_pos = drop_pos

        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True)

    def forward(self, M):
        """
            M : observations (batch_size, time_steps, observarions_channels)

            return: observations with sin cos encodning of angle (batch_size, time_series_latent_size)
        """
        q1, q2, dq1, dq2, u = torch.unbind(M, dim=-1)
        sq1 = torch.sin(q1)
        cq1 = torch.cos(q1)
        sq2 = torch.sin(q2)
        cq2 = torch.cos(q2)
        
        dq1 = dq1 / self.max_dq1
        dq2 = dq2 / self.max_dq2
        u = u / self.max_u
                
        if self.drop_pos:
            return torch.stack([dq1, dq2, u], dim=-1)
        
        if self.only_control:
            return torch.stack([u], dim=-1)
    
        if self.drop_control:
            return torch.stack([sq1, cq1, sq2, cq2, dq1, dq2], dim=-1)
        else:
            return torch.stack([sq1, cq1, sq2, cq2, dq1, dq2, u], dim=-1)
