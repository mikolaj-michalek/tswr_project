import numpy as np
from envs.drift360_env import Drift360Env

env = Drift360Env()
obs, info = env.reset()

reached_360 = False
after_360_steps = 0

for i in range(3000):
    if i < 40:
        action = np.array([0.0, 1.0])

    elif abs(env.yaw_unwrapped) < np.deg2rad(270):
        # kręcimy w lewo
        action = np.array([1.0, 1.0])

    elif np.deg2rad(270) < abs(env.yaw_unwrapped) < np.deg2rad(360):
        # kontra
        action = np.array([-1.0, 1.0])

    else:
        # wyjazd
        action = np.array([0.0, 0.7])

    obs, reward, terminated, truncated, info = env.step(action)

    if abs(env.yaw_unwrapped) >= 2 * np.pi and not reached_360:
        reached_360 = True
        print("360 reached")

        # opcjonalnie: zerujemy licznik obrotu, żeby nie liczył drugiego kółka
        # env.yaw_unwrapped = np.sign(env.yaw_unwrapped) * 2 * np.pi

    running = env.render()
    if not running:
        break

    if reached_360 and after_360_steps > 200:
        break

env.close()