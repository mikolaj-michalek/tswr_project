import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from racing_env.utils.obs_config import ObservationConfig


class DEFAULTS:
    TRACKS = ["icra_2023"]
    ACTION_SPACE = gym.spaces.Box(
        low=np.array([-1,] * 2),
        high=np.array([1] * 2),
        shape=(2,),
        dtype=np.float32,
    )
    OBS_CONFIG = "basic"