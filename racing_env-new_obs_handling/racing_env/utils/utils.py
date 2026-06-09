from typing import NamedTuple
import numpy as np
import torch
import gymnasium as gym


Track = NamedTuple(
    "Track",
    [
        ("s", np.ndarray),
        ("x", np.ndarray),
        ("y", np.ndarray),
        ("width", np.ndarray),
        ("curvature", np.ndarray),
        ("heading", np.ndarray),
    ],
)

def to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    If the input is already a NumPy array, it returns it unchanged.
    """
    if isinstance(tensor, np.ndarray) or tensor is None:
        return tensor
    elif isinstance(tensor, (list, tuple)):
        tensor = [to_numpy(x) if isinstance(x, (dict,)) else x for x in tensor]
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    elif isinstance(tensor, dict):
        return {key: to_numpy(value) for key, value in tensor.items()}
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")

def make_vec_space(space, num_envs):
    return gym.spaces.Box(
            low=np.tile(space.low[None], (num_envs, 1)),
            high=np.tile(space.high[None], (num_envs, 1)),
            shape=(num_envs, space.shape[0]))

             
from racetrack import LIST_OF_TRACKS

def get_available_tracks():
    return LIST_OF_TRACKS