import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.drift360_env import Drift360Env


os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = Drift360Env()
env = Monitor(env)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="models/",
    name_prefix="drift360_ppo",
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    tensorboard_log="logs/",
)

# model = PPO.load(
#     "models/drift360_ppo_fajny_v02",
#     env=env,
#     device="cpu"
# )

model.learn(
    total_timesteps=800_000,
    callback=checkpoint_callback,
    #reset_num_timesteps=False,
)

model.save("models/drift360_ppo_final")

env.close()