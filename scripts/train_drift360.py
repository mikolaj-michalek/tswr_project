import os
import numpy as np
import gymnasium as gym
import racing_env

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList


MODEL_DIR = "models/drift360_phase3"
LOG_DIR = "logs/drift360_phase3"

# startujemy od najlepszego starego modelu jazdy
LOAD_MODEL_PATH = "models/drift360_phase2/ppo_drift360_600000_steps"


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


class Drift360MetricsCallback(BaseCallback):
    """
    Dodatkowe logi do TensorBoard.
    Callback próbuje wyciągnąć dane z rewarda / env.
    Jak jakiegoś pola nie znajdzie, po prostu je pomija.
    """

    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _to_float(self, x):
        try:
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            x = np.asarray(x).reshape(-1)
            return float(x[0])
        except Exception:
            return None

    def _get_reward_obj(self):
        env = self.training_env.envs[0]

        candidates = [
            env,
            getattr(env, "vec_env", None),
            getattr(getattr(env, "vec_env", None), "env", None),
            getattr(getattr(env, "vec_env", None), "unwrapped", None),
        ]

        for obj in candidates:
            if obj is None:
                continue

            for attr in ["reward", "reward_fn", "_reward", "reward_function"]:
                reward_obj = getattr(obj, attr, None)
                if reward_obj is not None:
                    return reward_obj

        return None

    def _get_base_env(self):
        env = self.training_env.envs[0]

        if hasattr(env, "vec_env"):
            return env.vec_env

        return env

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        reward_obj = self._get_reward_obj()
        base_env = self._get_base_env()

        if reward_obj is not None:
            yaw_sum = self._to_float(getattr(reward_obj, "yaw_sum", None))
            phase = self._to_float(getattr(reward_obj, "phase", None))
            spin_steps = self._to_float(getattr(reward_obj, "spin_steps", None))
            cooldown = self._to_float(getattr(reward_obj, "cooldown", None))

            if yaw_sum is not None:
                self.logger.record("custom/yaw_sum_rad", yaw_sum)
                self.logger.record("custom/yaw_sum_deg", yaw_sum * 180.0 / np.pi)

            if phase is not None:
                self.logger.record("custom/phase", phase)

            if spin_steps is not None:
                self.logger.record("custom/spin_steps", spin_steps)

            if cooldown is not None:
                self.logger.record("custom/cooldown", cooldown)

        # próba logowania rzeczy ze środowiska
        for name in [
            "heading_diff",
            "signed_dist_to_centerline",
            "progress",
            "signed_dist_to_edge",
        ]:
            value = self._to_float(getattr(base_env, name, None))
            if value is not None:
                self.logger.record(f"custom/{name}", value)

        # prędkość z env.state, jeżeli jest dostępna
        state = getattr(base_env, "state", None)
        if state is not None:
            try:
                if hasattr(state, "detach"):
                    state_np = state.detach().cpu().numpy()
                else:
                    state_np = np.asarray(state)

                v_x = float(state_np.reshape(state_np.shape[0], -1)[0, 3])
                v_y = float(state_np.reshape(state_np.shape[0], -1)[0, 4])
                r = float(state_np.reshape(state_np.shape[0], -1)[0, 5])
                speed = float(np.sqrt(v_x ** 2 + v_y ** 2))

                self.logger.record("custom/v_x", v_x)
                self.logger.record("custom/v_y", v_y)
                self.logger.record("custom/r", r)
                self.logger.record("custom/speed", speed)
            except Exception:
                pass

        return True


def make_env():
    vec_env = gym.make_vec(
        "racing_env/SingleTrack-v0",
        num_envs=8,
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
        name_prefix="ppo_drift360_phase3",
    )

    metrics_callback = Drift360MetricsCallback(
        log_freq=1000,
    )

    callback = CallbackList([
        checkpoint_callback,
        metrics_callback,
    ])

    model = PPO.load(
        LOAD_MODEL_PATH,
        env=env,
        device="auto",
    )

    # phase3: zwiększona eksploracja
    model.ent_coef = 0.003

    # nowy katalog TensorBoard
    model.tensorboard_log = LOG_DIR

    model.learn(
        total_timesteps=4_000_000,
        callback=callback,
        tb_log_name="ppo_drift360_phase3_ent003",
        reset_num_timesteps=False,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_drift360_phase3_final"))
    env.close()


if __name__ == "__main__":
    main()