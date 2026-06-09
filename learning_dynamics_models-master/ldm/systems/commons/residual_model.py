import torch
from typing import List

from ldm.systems.mlp.mlp import Mlp



class ResidualModel(torch.nn.Module):
    def __init__(self, 
                 base_model: torch.nn.Module,
                 base_model_input_dim: int,
                 preprocessor: torch.nn.Module,
                 layer_sizes: List, # [input_dim, hidden1_dim, ..., output_dim]
                 activation: torch.nn.Module,
                ):
        super(ResidualModel, self).__init__()
        
        self.base_model = base_model
        self.base_model_input_dim = base_model_input_dim
        
        self.nn = Mlp(
            preprocessor=preprocessor,
            layer_sizes=layer_sizes,
            activation=activation
        )
                
    def forward(self, t, x, u):
        dx_base = self.base_model(t, x[..., :self.base_model_input_dim], u)[0]
        dx_residual = self.nn(t, x, u)[0]
        dx_base = torch.cat([dx_base, torch.zeros_like(x[..., self.base_model_input_dim:])], dim=-1)
        return dx_base + dx_residual, torch.zeros_like(x)  # return zero tire forces as placeholder

    def state_weights(self):
        return self.base_model.state_weights()