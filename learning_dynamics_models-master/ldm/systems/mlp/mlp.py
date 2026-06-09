import torch
from typing import List

class Mlp(torch.nn.Module):

    def __init__(self,
                 preprocessor: torch.nn.Module,
                 layer_sizes: List, # [input_dim, hidden1_dim, ..., output_dim]
                 activation: torch.nn.Module,
                 compile: bool = False,
                 *args, **kwargs) -> None:
        super(Mlp, self).__init__(*args, **kwargs)

        self.layer_sizes = list(layer_sizes)
        
        self.layer_count = len(layer_sizes)
        self.activation = activation
        self.preprocessor = preprocessor

        layers = []
        for i, (in_features, out_features) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            layers.append(torch.nn.Linear(in_features, out_features))
            # Add activation after each layer except the last one
            if i < len(self.layer_sizes) - 2:
                layers.append(self.activation)
        self.layers = torch.nn.Sequential(*layers)

        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=False)

    def forward(self, t, x, u):
        """
            t = [batch]
            x = [batch, state dim]
            u = [batch, action dim]
        """
        xu = torch.cat([x, u], dim=-1)
        # unsqueeze to get vector that can be multiplied with weights
        xu = self.preprocessor(xu)
        result = self.layers(xu)
        # return zero tire forces and slips as placeholders
        return result, torch.zeros_like(result), torch.zeros_like(result)
