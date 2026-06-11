import numpy as np
import gymnasium as gym
import racing_env
import pygame  # Zmieniamy keyboard na pygame

env = gym.make_vec(
    "racing_env/SingleTrack-v0",
    num_envs=1,
    max_episode_steps=1000,
    dt=0.02,
    frame_skip=5,
    tracks=["ml_wide_playground"],
    learned_model_config="racing_env-new_obs_handling/racing_env/envs/simulators/robot_models/ldm_models/pacejka.yml",
    reward_type="drift360",
    compile=False,
)

obs, info = env.reset()

env.start_render()

print("🏁 KLIKNIJ NA OKNO SYMULACJI, a następnie używaj W, A, S, D do sterowania. Naciśnij 'Q', aby wyjść.")

for i in range(10000):

    # Aktualizuje stan zdarzeń pygame (niezbędne, aby odczytać klawiaturę)
    pygame.event.pump()

    # Pobiera stan wszystkich klawiszy
    keys = pygame.key.get_pressed()

    steering = 0.0
    throttle = 0.0

    # --- OBSŁUGA KLAWIATURY (PYGAME) ---
    if keys[pygame.K_w]:
        throttle = 1.0
    elif keys[pygame.K_s]:
        throttle = -1.0

    if keys[pygame.K_a]:
        steering = 1.0
    elif keys[pygame.K_d]:
        steering = -1.0

    if keys[pygame.K_q]:
        print("Zakończono jazdę.")
        break

    action = np.array([[steering, throttle]], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    if i % 20 == 0:
        print(f"Krok: {i}, Akcja: {action[0]}")

    env.render([0])

env.close()