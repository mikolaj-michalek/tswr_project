import numpy as np
import gymnasium as gym
import racing_env

env = gym.make_vec(
    "racing_env/SingleTrack-v0",
    num_envs=1,
    max_episode_steps=1000,
    dt=0.02,
    frame_skip=5,
    tracks=["ml_track"],
    learned_model_config="racing_env-new_obs_handling/racing_env/envs/simulators/robot_models/ldm_models/pacejka.yml",
    reward_type="drift360",
    compile=False,
)

obs, info = env.reset()

env.start_render()

for i in range(10000):

    action = np.array([[0.8, 1.0]], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    if i % 20 == 0:
        print(i, reward)

    env.render([0])

env.close()