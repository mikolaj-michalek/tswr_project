import numpy as np

from stable_baselines3 import PPO
from envs.drift360_env import Drift360Env


env = Drift360Env()
#model = PPO.load("models/drift360_ppo_fajny_v01")
#model = PPO.load("models/drift360_ppo_fajny_v02")
#model = PPO.load("models/drift360_ppo_fajny_v03") # dla V_0 = 70.0
model = PPO.load("models/drift360_ppo_fajny_v04") # dla V_0 = 100.0

#model = PPO.load("models/drift360_ppo_100000_steps")
#model = PPO.load("models/drift360_ppo_200000_steps")
#model = PPO.load("models/drift360_ppo_300000_steps")
#model = PPO.load("models/drift360_ppo_400000_steps")
#model = PPO.load("models/drift360_ppo_500000_steps")
#model = PPO.load("models/drift360_ppo_600000_steps")
#model = PPO.load("models/drift360_ppo_700000_steps")
#model = PPO.load("models/drift360_ppo_800000_steps")
# model = PPO.load("models/drift360_ppo_900000_steps")
# model = PPO.load("models/drift360_ppo_1000000_steps")
# model = PPO.load("models/drift360_ppo_1100000_steps")
# model = PPO.load("models/drift360_ppo_1200000_steps")
# model = PPO.load("models/drift360_ppo_1300000_steps")
# model = PPO.load("models/drift360_ppo_1400000_steps")


#model = PPO.load("models/drift360_ppo_final")

obs, info = env.reset()

for i in range(2000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    running = env.render()
    if not running:
        break

    if terminated or truncated:
        print("Episode finished")
        print(f"yaw: {np.rad2deg(env.yaw_unwrapped):.1f} deg")
        print(f"x: {env.x:.2f}, y: {env.y:.2f}")
        print(f"vx: {env.vx:.2f}, vy: {env.vy:.2f}, r: {env.r:.2f}")
        print(f"done reason: {env.done_reason}")
        break

env.close()