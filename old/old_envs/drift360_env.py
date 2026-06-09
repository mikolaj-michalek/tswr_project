import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import math


class Drift360Env(gym.Env):
    def __init__(self):
        super().__init__()

        self.dt = 0.02
        self.max_steps = 2500

        self.m = 1200.0
        self.Iz = 1500.0
        self.lf = 1.2
        self.lr = 1.4

        self.Cf = 3500.0
        self.Cr = 2500.0

        self.max_force = 12000.0
        self.drag = 100.0
        self.max_delta = 0.7

        self.start_x = -30.0
        self.start_y = 0.0

        self.target_radius = 18.0
        self.max_radius = 450.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.start_x
        self.y = self.start_y
        self.yaw = 0.0
        self.yaw_unwrapped = 0.0

        self.vx = 100.0
        self.vy = 0.0
        self.r = 0.0

        self.step_count = 0

        self.completed_360 = False
        self.after_360_steps = 0

        self.done_reason = ""

        # w reset()
        self.prev_delta = 0.0
        self.steering_reversal_done = False

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        delta = float(action[0]) * self.max_delta
        throttle = float(action[1])

        prev_yaw_progress = abs(self.yaw_unwrapped)

        vx_safe = max(abs(self.vx), 0.5)

        alpha_f = np.arctan2(self.vy + self.lf * self.r, vx_safe) - delta
        alpha_r = np.arctan2(self.vy - self.lr * self.r, vx_safe)

        Fyf = -self.Cf * alpha_f
        Fyr = -self.Cr * alpha_r
        Fx = throttle * self.max_force - self.drag * self.vx

        vx_dot = (Fx - Fyf * np.sin(delta)) / self.m + self.vy * self.r
        vy_dot = (Fyr + Fyf * np.cos(delta)) / self.m - self.vx * self.r
        r_dot = (self.lf * Fyf * np.cos(delta) - self.lr * Fyr) / self.Iz

        self.vx += vx_dot * self.dt
        self.vy += vy_dot * self.dt
        self.r += r_dot * self.dt

        self.yaw += self.r * self.dt
        self.yaw_unwrapped += self.r * self.dt

        self.x += (self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)) * self.dt
        self.y += (self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)) * self.dt

        beta = np.arctan2(self.vy, max(abs(self.vx), 0.5))
        yaw_progress = abs(self.yaw_unwrapped)
        delta_yaw_progress = yaw_progress - prev_yaw_progress

        distance_from_start = np.sqrt(
            (self.x - self.start_x) ** 2 + (self.y - self.start_y) ** 2
        )

        speed = np.sqrt(self.vx**2 + self.vy**2)

        reward = 0.0
        terminated = False

        road_half_width = 40.0
        road_error = abs(self.y)

        if not self.completed_360 and not self.steering_reversal_done:
            if self.prev_delta < -0.3 and delta > 0.3:
                reward += 250.0
                self.steering_reversal_done = True

        self.prev_delta = delta

        # Faza 1/2: przed ukończeniem 360
        if not self.completed_360:
            # nagroda za postęp obrotu
            reward += 90.0 * max(delta_yaw_progress, 0.0)

            # zachęta do utrzymania prędkości
            reward += 0.6 * min(speed, 18.0)

            # zachęta do poślizgu, ale ograniczona
            reward += 4.0 * min(abs(beta), 1.0)

            # ciasna przestrzeń manewru
            if distance_from_start <= self.target_radius:
                reward += 6.0
            else:
                reward -= 0.7 * (distance_from_start - self.target_radius)

            # kara za stanie / zbyt wolną jazdę
            if speed < 5.0:
                reward -= 15.0

            # kara za czas
            reward -= 0.1

            # wykrycie pierwszego 360
            if yaw_progress >= 2 * np.pi:
                self.completed_360 = True
                self.after_360_steps = 0

                reward += 700.0
                reward += 25.0 * min(speed, 18.0)
                reward -= 8.0 * max(0.0, distance_from_start - self.target_radius)

            # trzymanie się środka drogi przed i w trakcie 360
            reward += 2.0 * max(0.0, 1.0 - road_error / road_half_width)

            # kara za wyjazd poza drogę
            if road_error > road_half_width:
                reward -= 8.0 * (road_error - road_half_width)

        # Faza 3: po 360 ma jechać dalej, a nie robić drugie koło
        else:
            self.after_360_steps += 1

            reward += 2.0 * max(self.vx, 0.0)

            # bardzo mocno karz dalsze obracanie
            reward -= 250.0 * abs(self.r)

            # bonus tylko gdy naprawdę stabilizuje auto
            if abs(self.r) < 0.4 and self.vx > 5.0:
                reward += 120.0

            # bonus za utrzymanie stabilizacji przez chwilę
            if abs(self.r) < 0.4 and abs(beta) < 0.5 and self.vx > 5.0:
                reward += 200.0

            # kara za zbliżanie się do drugiego obrotu
            extra_spin = yaw_progress - 2 * np.pi
            if extra_spin > 0:
                reward -= 500.0 * extra_spin

            if yaw_progress > 2.15 * np.pi:
                reward -= 1500.0
                terminated = True
                self.done_reason = "second spin"

            if self.after_360_steps > 200:
                reward += 800.0
                terminated = True
                self.done_reason = "success"

            # po 360 mocno nagradzamy powrót na środek drogi
            reward += 6.0 * max(0.0, 1.0 - road_error / road_half_width)

            # duża kara za jazdę poza drogą po manewrze
            if road_error > road_half_width:
                reward -= 18.0 * (road_error - road_half_width)

        # Twarde ograniczenia przestrzeni
        if distance_from_start > self.max_radius:
            reward -= 700.0
            terminated = True
            self.done_reason = "too far"

        # Zabezpieczenia numeryczne
        if abs(self.vx) > 500.0 or abs(self.vy) > 350.0:
            reward -= 300.0
            terminated = True
            self.done_reason = "too fast"

        if not np.isfinite(reward):
            reward = -1000.0
            terminated = True

        if road_error > road_half_width:
            reward -= 1000.0
            terminated = True
            self.done_reason = "left road"

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        beta = np.arctan2(self.vy, max(abs(self.vx), 0.5))

        distance_from_start = np.sqrt(
            (self.x - self.start_x) ** 2 + (self.y - self.start_y) ** 2
        )

        speed = np.sqrt(self.vx**2 + self.vy**2)

        return np.array([
            self.vx / 30.0,
            self.vy / 30.0,
            self.r / 8.0,
            self.yaw_unwrapped / (2 * np.pi),
            beta,
            distance_from_start / self.max_radius,
            speed / 30.0,
            float(self.completed_360),
        ], dtype=np.float32)

    def render(self):
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen_height = 1000
            self.screen_width = 3000
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Drift 360 Env")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.close()
                    return False

        self.screen.fill((255, 255, 255))

        scale = 6
        cx = 180
        cy = self.screen_height // 2

        car_length = 40
        car_width = 22

        road_width_m = 40.0
        road_top = int(cy - (road_width_m * scale) / 2)
        road_height = int(road_width_m * scale)

        pygame.draw.rect(
            self.screen,
            (210, 210, 210),
            pygame.Rect(0, road_top, self.screen_width, road_height),
        )

        # linia środka drogi
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (0, cy),
            (self.screen_width, cy),
            2,
        )

        car_x = int(cx + self.x * scale)
        car_y = int(cy - self.y * scale)

        car_length = 40
        car_width = 22

        corners = np.array([
            [car_length / 2, car_width / 2],
            [car_length / 2, -car_width / 2],
            [-car_length / 2, -car_width / 2],
            [-car_length / 2, car_width / 2],
        ])

        rot = np.array([
            [math.cos(-self.yaw), -math.sin(-self.yaw)],
            [math.sin(-self.yaw), math.cos(-self.yaw)],
        ])

        rotated = corners @ rot.T
        points = [(car_x + px, car_y + py) for px, py in rotated]

        pygame.draw.polygon(self.screen, (50, 100, 220), points)

        font = pygame.font.SysFont(None, 28)

        phase = "after 360" if self.completed_360 else "before 360"

        text = font.render(
            f"yaw: {np.rad2deg(self.yaw_unwrapped):.1f} deg | "
            f"vx: {self.vx:.1f} | vy: {self.vy:.1f} | r: {self.r:.2f} | {phase}",
            True,
            (0, 0, 0),
        )
        self.screen.blit(text, (20, 20))

        pygame.display.flip()
        self.clock.tick(50)

        return True

    def close(self):
        if hasattr(self, "screen"):
            pygame.quit()
            del self.screen