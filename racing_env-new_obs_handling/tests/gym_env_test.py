from time import sleep
import torch
import numpy as np
import gymnasium as gym
from racing_env.envs.gymnasium import SingleTrackVecEnv


#n_envs = 20
n_envs = 10
n_steps = 1000
dt = 0.02
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

#vec_env = gym.make_vec("Pendulum-v1", num_envs=n_envs)
#a = 0

# vec_env = gym.make_vec(
#     "racing_env/SingleTrack-v0",
#     num_envs=n_envs,
#     n_steps=n_steps,
#     dt=dt,
# )

vec_env = gym.make_vec(
    "racing_env/SingleTrack-v0",
    num_envs=n_envs,
    max_episode_steps=n_steps,
    dt=dt,
    learned_model_config="racing_env/envs/simulators/robot_models/ldm_models/pacejka.yml",
    compile=False,
)

a = vec_env.reset()
for i in range(n_steps):
    print(f"Step {i + 1}/{n_steps}")
    a = np.random.randn(n_envs, 2).astype(np.float32)
    vec_env.step(a)
    vec_env.render(idxs=[0, 1])