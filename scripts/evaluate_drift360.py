import time
import numpy as np
import gymnasium as gym
import racing_env

from stable_baselines3 import PPO

# TEST
# MODEL_PATH = "models/drift360_phase6/test_ppo_drift360_phase6_150000_steps" 
# MODEL_PATH = "models/drift360_phase4/test_ppo_drift360_phase4_1050000_steps"

# BEST
MODEL_PATH = "models/drift360_phase6/best_ppo_drift360_phase6_300000_steps"  
TRACK = "ml_track"


def unwrap_obs(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v)[0] for k, v in obs.items()}
    return np.asarray(obs)[0]


def wrap_action(action):
    action = np.asarray(action, dtype=np.float32)
    return action.reshape(1, -1)


def get_reward_debug(env):
    candidates = [
        getattr(env, "reward", None),
        getattr(env, "reward_fn", None),
        getattr(env, "_reward", None),
        getattr(env, "unwrapped", None),
    ]

    for c in candidates:
        if c is not None and hasattr(c, "last_debug_info"):
            return c.last_debug_info

    if hasattr(env, "envs"):
        inner = env.envs[0]
        for name in ["reward", "reward_fn", "_reward"]:
            c = getattr(inner, name, None)
            if c is not None and hasattr(c, "last_debug_info"):
                return c.last_debug_info

    for name in dir(env):
        try:
            obj = getattr(env, name)
            if hasattr(obj, "last_debug_info"):
                return obj.last_debug_info
        except Exception:
            pass

    return None


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

    last_phase = None
    best_yaw_episode = 0.0

    for step in range(5000):

        action, _ = model.predict(
            unwrap_obs(obs),
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(wrap_action(action))

        reward_value = float(np.asarray(reward).reshape(-1)[0])

        done = bool(
            np.asarray(terminated).reshape(-1)[0]
            or np.asarray(truncated).reshape(-1)[0]
        )

        episode_reward += reward_value
        episode_steps += 1

        state = env.state

        yaw = float(state[0, 2])
        vx = float(state[0, 3])
        vy = float(state[0, 4])
        yaw_rate = float(state[0, 5])
        speed = np.sqrt(vx**2 + vy**2)

        debug = get_reward_debug(env)

        if debug is not None:
            phase = debug["phase_name"][0]
            yaw_deg = debug["yaw_deg"][0]
            best_yaw_deg = debug["best_yaw_deg"][0]
            beta_deg = debug["beta_deg"][0]
            drift_steps = debug["drift_steps"][0]
            recover_steps = debug["recover_steps"][0]
            reward_from_debug = debug["reward"][0]

            best_yaw_episode = max(best_yaw_episode, best_yaw_deg)

            if phase != last_phase:
                print(
                    "\n"
                    f"PHASE CHANGE | ep={episode_id} | "
                    f"step={step} | "
                    f"{last_phase} -> {phase} | "
                    f"yaw={yaw_deg:.1f} deg | "
                    f"best={best_yaw_deg:.1f} deg | "
                    f"speed={speed:.2f} | "
                    f"beta={beta_deg:.1f} deg"
                )
                last_phase = phase

        else:
            phase = "UNKNOWN"
            yaw_deg = 0.0
            best_yaw_deg = 0.0
            beta_deg = 0.0
            drift_steps = 0
            recover_steps = 0
            reward_from_debug = reward_value

        if step % 20 == 0:
            print(
                f"ep={episode_id:03d} | "
                f"step={step:05d} | "
                f"phase={phase:<7} | "
                f"reward={reward_value:+.4f} | "
                f"ep_reward={episode_reward:+.2f} | "
                f"yaw_sum={yaw_deg:6.1f} deg | "
                f"best_yaw={best_yaw_deg:6.1f} deg | "
                f"speed={speed:+.2f} | "
                f"vx={vx:+.2f} | "
                f"vy={vy:+.2f} | "
                f"beta={beta_deg:+.1f} deg | "
                f"r={yaw_rate:+.2f} | "
                f"drift_steps={drift_steps:03d} | "
                f"recover_steps={recover_steps:03d} | "
                f"done={done} | "
                f"action={np.asarray(action)}"
            )

        env.render([0])
        time.sleep(0.01)

        if done:
            print(
                "\n"
                f"EPISODE {episode_id} END | "
                f"steps={episode_steps} | "
                f"sum_reward={episode_reward:+.2f} | "
                f"best_yaw={best_yaw_episode:.1f} deg"
                "\n"
            )

            episode_id += 1
            episode_reward = 0.0
            episode_steps = 0
            best_yaw_episode = 0.0
            last_phase = None

            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()