import time
import numpy as np
import gymnasium as gym
import racing_env

from stable_baselines3 import PPO


MODEL_PATH = "models/drift360/ppo_drift360_500000_steps"
# MODEL_PATH = "models/drift360/ppo_drift360_final"

TRACK = "ml_track"


def unwrap_obs(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v)[0] for k, v in obs.items()}
    return np.asarray(obs)[0]


def wrap_action(action):
    action = np.asarray(action, dtype=np.float32)
    return action.reshape(1, -1)


def main():
    env = gym.make_vec(
        "racing_env/SingleTrack-v0",
        num_envs=1,
        max_episode_steps=1000,
        dt=0.02,
        frame_skip=5,
        tracks=[TRACK],
        learned_model_config="racing_env-new_obs_handling/racing_env/envs/simulators/robot_models/ldm_models/pacejka.yml",
        reward_type="drift360",
        observation_config="basic",
        initializer="zero",
        compile=False,
    )

    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()

    env.start_render()

    episode_reward = 0.0
    episode_steps = 0
    episode_id = 0

    for step in range(5000):
        action, _ = model.predict(
            unwrap_obs(obs),
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(wrap_action(action))

        r = float(np.asarray(reward).reshape(-1)[0])
        done = bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])

        episode_reward += r
        episode_steps += 1

        if step % 20 == 0:
            print(
                f"step={step:05d} | "
                f"reward={r:+.4f} | "
                f"episode_reward={episode_reward:+.2f} | "
                f"done={done} | "
                f"action={np.asarray(action)}"
            )

        env.render([0])
        time.sleep(0.01)

        if done:
            print(
                f"EPISODE {episode_id} END | "
                f"steps={episode_steps} | "
                f"sum_reward={episode_reward:+.2f}"
            )

            episode_id += 1
            episode_reward = 0.0
            episode_steps = 0

            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()