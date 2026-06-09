import torch
from typing import List


class MlpWithHistory(torch.nn.Module):
    """
    MLP dynamics model that receives the N most recent observations
    (history) concatenated with the current state and control input.

    Expected call signature (compatible with RolloutModelWithHistory):
        forward(t, history, x, u) -> (dx, tire_forces, slips)

    where
        t       : (1,)                              – current time
        history : (batch_size, history_len, obs_dim) – past observations
        x       : (batch_size, state_dim)            – current state
        u       : (batch_size, control_dim)          – current control

    The history is flattened and concatenated with (x, u) before being
    passed through the MLP.  The preprocessor is applied to (x, u) only;
    the history is appended afterwards so that normalisation/preprocessing
    stays the same as for the plain MLP.
    """

    def __init__(
        self,
        preprocessor: torch.nn.Module,
        layer_sizes: List[int],  # [input_dim, hidden1_dim, ..., output_dim]
        activation: torch.nn.Module,
        history_len: int,
        obs_dim: int,
        compile: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.history_len = history_len
        self.obs_dim = obs_dim
        self.preprocessor = preprocessor
        self.activation = activation

        # Augment the first layer to also accept flattened history
        aug_layer_sizes = list(layer_sizes)
        aug_layer_sizes[0] = layer_sizes[0] + history_len * obs_dim

        self.layer_sizes = aug_layer_sizes

        layers = []
        for i, (in_f, out_f) in enumerate(
            zip(aug_layer_sizes[:-1], aug_layer_sizes[1:])
        ):
            layers.append(torch.nn.Linear(in_f, out_f))
            if i < len(aug_layer_sizes) - 2:
                layers.append(self.activation)
        self.layers = torch.nn.Sequential(*layers)

        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=False)

    def forward(self, t, history, x, u):
        """
        Args:
            t       : (1,)
            history : (batch_size, history_len, obs_dim)
            x       : (batch_size, state_dim)
            u       : (batch_size, control_dim)

        Returns:
            dx         : (batch_size, state_dim)
            tire_forces: zeros placeholder
            slips      : zeros placeholder
        """
        xu = torch.cat([x, u], dim=-1)
        xu = self.preprocessor(xu)  # (batch_size, preprocessed_dim)

        hist = self.preprocessor(history)
        hist_flat = hist.reshape(hist.shape[0], -1)  # (batch_size, history_len * obs_dim)

        xu_hist = torch.cat([xu, hist_flat], dim=-1)

        result = self.layers(xu_hist)
        return result, torch.zeros_like(result), torch.zeros_like(result)
