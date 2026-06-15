import os
import numpy as np
import gymnasium as gym
import racing_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

MODEL_DIR = "models/drift360_phase6"
LOG_DIR = "logs/drift360_phase6"

#LOAD_MODEL_PATH = "models/drift360_phase6/ppo_drift360_phase6_400000_steps"  # Set to None to train from scratch

class SingleVecToGymWrapper(gym.Env):
    def __init__(self, vec_env):
        super().__init__()
        self.vec_env = vec_env
        self.observation_space = vec_env.single_observation_space
        self.action_space = vec_env.single_action_space

    def _unwrap_obs(self, obs):
        if isinstance(obs, dict):
            return {k: np.asarray(v)[0] for k, v in obs.items()}
        return np.asarray(obs)[0]

    def reset(self, seed=None, options=None):
        obs, info = self.vec_env.reset(seed=seed, options=options)
        return self._unwrap_obs(obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape(1, -1)

        obs, reward, terminated, truncated, info = self.vec_env.step(action)

        reward = float(np.asarray(reward).reshape(-1)[0])
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])

        if isinstance(info, dict):
            info = {
                k: v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v
                for k, v in info.items()
            }
        else:
            info = {}

        return self._unwrap_obs(obs), reward, terminated, truncated, info

    def close(self):
        self.vec_env.close()


def make_env():
    vec_env = gym.make_vec(
        "racing_env/SingleTrack-v0",
        num_envs=1,
        max_episode_steps=1000,
        dt=0.02,
        frame_skip=5,
        tracks=["ml_track"],
        learned_model_config="racing_env-new_obs_handling/racing_env/envs/simulators/robot_models/ldm_models/pacejka.yml",
        reward_type="drift360",
        observation_config="basic",
        initializer="zero",
        compile=False,
    )

    return SingleVecToGymWrapper(vec_env)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_drift360_phase6",
    )

    # ------------------------------------------------------
    #                       CREATE
    # ------------------------------------------------------

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4, 
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu",
    )

    # ------------------------------------------------------
    #                       LOAD
    # ------------------------------------------------------

    # model = PPO.load(
    #     LOAD_MODEL_PATH,
    #     env=env,
    #     device="cpu",
    # )
    # model.tensorboard_log = LOG_DIR

    model.learn(
        total_timesteps=5_000_000,
        callback=checkpoint_callback,
        tb_log_name="ppo_drift360_phase6",
        reset_num_timesteps=False,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_drift360_phase6_final"))
    env.close()

if __name__ == "__main__":
    main()