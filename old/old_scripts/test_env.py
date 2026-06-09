import numpy as np
import matplotlib.pyplot as plt

from envs.drift360_env import Drift360Env


env = Drift360Env()
obs, info = env.reset()

xs = []
ys = []
yaws = []
vxs = []
vys = []
rs = []

for i in range(1000):
    action = np.array([0.8, 1.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    xs.append(env.x)
    ys.append(env.y)
    yaws.append(env.yaw_unwrapped)
    vxs.append(env.vx)
    vys.append(env.vy)
    rs.append(env.r)

    if terminated or truncated:
        break

print(f"steps: {i}")
print(f"yaw deg: {np.rad2deg(env.yaw_unwrapped):.1f}")
print(f"x: {env.x:.2f}, y: {env.y:.2f}")
print(f"vx: {env.vx:.2f}, vy: {env.vy:.2f}, r: {env.r:.2f}")

plt.figure()
plt.plot(xs, ys)
plt.axis("equal")
plt.grid(True)
plt.title("Trajectory")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()

plt.figure()
plt.plot(np.rad2deg(yaws))
plt.grid(True)
plt.title("Yaw progress")
plt.xlabel("step")
plt.ylabel("yaw [deg]")
plt.show()