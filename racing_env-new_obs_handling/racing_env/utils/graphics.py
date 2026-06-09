import math
import pygame
import numpy as np
import torch
import time
import logging
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt

from racing_env.utils.state_wrapper import StateWrapper
from racing_env.utils.constants import CAR
from racing_env.utils.utils import Track


def rect(x, y, angle, w, h, scale=10):
    x = x * scale
    y = y * scale
    w = w * scale
    h = h * scale

    return [
        translate(x, y, angle, -w / 2, h / 2),
        translate(x, y, angle, w / 2, h / 2),
        translate(x, y, angle, w / 2, -h / 2),
        translate(x, y, angle, -w / 2, -h / 2),
    ]


def translate(x, y, angle, px, py):
    x1 = x + px * math.cos(angle) - py * math.sin(angle)
    y1 = y + px * math.sin(angle) + py * math.cos(angle)
    return [int(x1), int(y1)]


def tensor_to_rdylgn(normalized_tensor):
    normalized_numpy = normalized_tensor.cpu().numpy()
    colormap = cm['RdYlGn']
    colored_np = colormap(normalized_numpy)[..., :3]
    return colored_np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 100, 0)
RED = (255, 0, 0)
GREY = (200, 200, 200)


class SceneRenderer:
    def __init__(
        self,
        vehicle_params: torch.Tensor,
        track_x,
        track_y,
        track_width,
        scale: float = 80 / 1,  # px per meter
        dt: float = 0.02,
        friction_map=None,
    ):
        self.vehicle_params = vehicle_params
        self.track = Track(
            s=None,
            x=track_x,
            y=track_y,
            width=track_width,
            curvature=None,
            heading=None,
        )
        self.scale = scale
        self.clock = pygame.time.Clock()

        self.friction_map = friction_map[0]

        self.dt = dt
        # log now tim
        self.last_render = time.time()

        self.min_friction = 0.65
        self.max_friction = 0.95

        self.track_min_x = np.min(self.track.x).item()
        self.track_max_x = np.max(self.track.x).item()
        self.track_min_y = np.min(self.track.y).item()
        self.track_max_y = np.max(self.track.y).item()

        self.screen_width = int((self.track_max_x - (self.track_min_x) + 2) * self.scale)
        self.screen_height = int((self.track_max_y - (self.track_min_y) + 2) * self.scale)

        # add pading to the track
        self.track_min_x -= 1
        self.track_min_y -= 1

        self._track_surface = None
        self._rendering_started = False

        logging.info("SceneRenderer initalized")

    def start_render(self):
        self._rendering_started = True
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.scale = 80 / 1  # px per meter
        self._cache_track()
        logging.info("Rendering started")
        print("Rendering started")

    def render(self, idxs: list, closest_idx: int, state: torch.tensor,
               trajectories: torch.tensor = None, costs: torch.tensor = None,
               lidar_data: torch.tensor = None):

        if not self._rendering_started:
            self.start_render()

        self.draw_track()
        self.draw_grid()

        if trajectories is not None:
            self.draw_trajectories(trajectories, costs)

        for idx in idxs:
            lidar_rays = lidar_data[idx] if lidar_data is not None else None
            self.render_car(idx, closest_idx[idx], state[idx], lidar_rays)

        now = time.time()
        diff = now - self.last_render
        offset = self.dt - diff
        wait = max(0, offset)
        if offset < 0: 
            logging.error(f"Rendering took too long: {diff}")
        time.sleep(wait)
        self.last_render = time.time()


        # flip the screen
        flipped_screen = pygame.transform.flip(self.screen, False, True)
        self.screen.blit(flipped_screen, (0, 0))

        self.draw_text()

        pygame.display.flip()
        self.clock.tick(0)

    def _cache_track(self):
        self._track_surface = pygame.Surface((self.screen_width, self.screen_height))
        self._track_surface.fill(WHITE)

        color_map = [GREY] * len(self.track.x)

        friction_map = self.friction_map.detach().cpu().numpy() if isinstance(self.friction_map, torch.Tensor) else self.friction_map
        if friction_map is not None:
            self.min_friction = round(min(friction_map[:self.track.x.shape[0]]), 2)
            self.max_friction = round(max(friction_map[:self.track.x.shape[0]]), 2)
            range_friction = self.max_friction - self.min_friction
            for i in range(len(self.track.x)):
                friction = friction_map[i]
                color = int((friction - self.min_friction) / (range_friction + 1e-8) * 100 + 100)
                color = max(0, min(color, 255))
                color_map[i] = (
                    color,
                    color,
                    color,
                )
                
        for i in range(len(self.track.x) - 1):
            x = self.track.x[i] - (self.track_min_x)
            y = self.track.y[i] - (self.track_min_y)
            w = self.track.width[i]

            pygame.draw.circle(
                self._track_surface,
                color_map[i],
                (int(x * self.scale), int(y * self.scale)),
                int(w * self.scale) / 2,  # radius is width / 2
                0,
            )

        for i in range(len(self.track.x) - 1):
            x = self.track.x[i] - (self.track_min_x)
            y = self.track.y[i] - (self.track_min_y)

            if i % 10 == 0:
                # write index of the point
                font = pygame.font.Font(None, 20)
                text = font.render(str(int(i / 10)), True, BLACK)
                self._track_surface.blit(
                    text, (int(x * self.scale), int(y * self.scale))
                )

        # add color legend at the bottom
        for i in range(256):
            color = int(i / 1.25)
            pygame.draw.rect(
                self._track_surface,
                (color, color, color),
                (i * 2, self.screen_height - 20, 2, 20),
            )

    def draw_track(self):
        if self._track_surface is None:
            self._cache_track()
        self.screen.blit(self._track_surface, (0, 0))

    def draw_grid(self):
        # draw grid 1m x 1m
        x_lines = np.arange(0, self.screen_width, 1 * self.scale)
        y_lines = np.arange(0, self.screen_height, 1 * self.scale)

        for x in x_lines:
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, self.screen_height), 1)

        for y in y_lines:
            pygame.draw.line(self.screen, BLACK, (0, y), (self.screen_width, y), 1)

    def draw_trajectories(self, trajectories, costs=None):
        trajectories = trajectories.detach().cpu().numpy()
        colors = None
        if costs is not None:
            rewards = -costs
            rng = (-100., 100.)
            rewards = torch.clip(rewards, rng[0], rng[1])
            rewards_normalized = (rewards - rng[0]) / (rng[1] - rng[0])
            colors = 255 * tensor_to_rdylgn(rewards_normalized)
        for i in range(len(trajectories)):
            #pygame.draw.lines(self.screen, RED, False, trajectories[i], 2)
            traj = [((x[0].item() - self.track_min_x) * self.scale, (x[1].item() - self.track_min_y) * self.scale)
                    for x in trajectories[i]]
            pygame.draw.lines(self.screen, colors[i] if colors is not None else RED, False, traj, 2)

    def render_car(self, idx, closest_idx, state, lidar_rays=None):
        rendered_state = StateWrapper(state)
        x = rendered_state.x - (self.track_min_x)
        y = rendered_state.y - (self.track_min_y)
        yaw = rendered_state.yaw
        delta = rendered_state.delta

        # vehicle_params layout: [m, g, I_z, L, lr, ...] (VehicleParameters order)
        L = self.vehicle_params[idx, 3].item()
        lr = self.vehicle_params[idx, 4].item()
        lf = L - lr

        # --- Dimensions ---
        car_length = lf + lr
        car_width = CAR.WIDTH
        wheel_len = car_length * 0.18
        wheel_width = wheel_len * 0.4

        # --- Math Helpers ---
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        # Geometric center offset from CoG (CoG != rectangle center when lf != lr)
        center_offset = (lf - lr) / 2
        body_x = x + center_offset * cos_yaw
        body_y = y + center_offset * sin_yaw

        # --- Shapes ---
        
        # 1. Car Body
        body = rect(body_x, body_y, yaw, car_length, car_width, self.scale)

        # 2. Wheel Positions (Relative to CoG (x,y))
        hw = car_width / 2

        fl_x = x + lf * cos_yaw - hw * sin_yaw
        fl_y = y + lf * sin_yaw + hw * cos_yaw
        fl_wheel = rect(fl_x, fl_y, yaw + delta, wheel_len, wheel_width, self.scale)

        fr_x = x + lf * cos_yaw + hw * sin_yaw
        fr_y = y + lf * sin_yaw - hw * cos_yaw
        fr_wheel = rect(fr_x, fr_y, yaw + delta, wheel_len, wheel_width, self.scale)

        rl_x = x - lr * cos_yaw - hw * sin_yaw
        rl_y = y - lr * sin_yaw + hw * cos_yaw
        rl_wheel = rect(rl_x, rl_y, yaw, wheel_len, wheel_width, self.scale)

        rr_x = x - lr * cos_yaw + hw * sin_yaw
        rr_y = y - lr * sin_yaw - hw * cos_yaw
        rr_wheel = rect(rr_x, rr_y, yaw, wheel_len, wheel_width, self.scale)
        
        # Center of Gravity Indicator
        center = rect(x, y, yaw, car_width/4, car_width/4, self.scale)

        # --- Drawing ---
        
        # Draw Body (Blue)
        pygame.draw.polygon(self.screen, BLUE, body)
        
        # Draw Wheels (Black or Dark Grey)
        wheel_color = (40, 40, 40)
        pygame.draw.polygon(self.screen, wheel_color, fl_wheel) # FL
        pygame.draw.polygon(self.screen, wheel_color, fr_wheel) # FR
        pygame.draw.polygon(self.screen, wheel_color, rl_wheel) # RL
        pygame.draw.polygon(self.screen, wheel_color, rr_wheel) # RR

        # Draw CoG (Green)
        pygame.draw.polygon(self.screen, GREEN, center)

        # --- Debug Visuals (Track points, Lines, Lidar) ---

        x_point = self.track.x[closest_idx] - (self.track_min_x)
        y_point = self.track.y[closest_idx] - (self.track_min_y)
        pygame.draw.circle(
            self.screen,
            RED,
            (int(x_point * self.scale), int(y_point * self.scale)),
            5,
            0,
        )

        pygame.draw.line(
            self.screen,
            BLACK,
            (int(x * self.scale), int(y * self.scale)),
            (int(x_point * self.scale), int(y_point * self.scale)),
            2,
        )

        # render every 5th track point from closest_idx
        for i in range(0, 70, 10):
            ponint_idx = (closest_idx + i) % len(self.track.x)
            x_point = self.track.x[ponint_idx] - (self.track_min_x)
            y_point = self.track.y[ponint_idx] - (self.track_min_y)
            pygame.draw.circle(
                self.screen,
                BLUE,
                (int(x_point * self.scale), int(y_point * self.scale)),
                2,
                0,
            )

        if lidar_rays is not None:
            num_rays = len(lidar_rays)
            angle_step = (2 * math.pi) / num_rays
            
            # Downsample for visualization if too many rays
            skip = 1 # max(1, num_rays // 30) 
            
            start_pos = (int(x * self.scale), int(y * self.scale))
            
            # Use numpy for easier iteration if needed, assuming tensor here
            rays_np = lidar_rays
            
            for i in range(0, num_rays, skip):
                dist = rays_np[i]
                if dist > 0.1: # Don't draw tiny rays
                    # Angle relative to car + car's yaw
                    # Adjust sign (+/-) depending on your Lidar rotation logic vs Pygame
                    ray_angle = yaw - (i * angle_step) 
                    
                    end_x = x + dist * math.cos(ray_angle)
                    end_y = y + dist * math.sin(ray_angle)
                    
                    end_pos = (int(end_x * self.scale), int(end_y * self.scale))
                    
                    # Draw a thin green line
                    pygame.draw.line(self.screen, (0, 255, 0), start_pos, end_pos, 1)

    def draw_text(self):
        y_offset = 30

        font = pygame.font.Font(None, 20)
        text = font.render(str(self.min_friction), True, BLACK)
        self.screen.blit(text, (5, y_offset))

        text = font.render(str(self.max_friction), True, BLACK)
        self.screen.blit(text, (self.screen_width-25, y_offset))

        text = font.render(str(round((self.max_friction + self.min_friction) / 2, 2)), True, BLACK)
        self.screen.blit(text, (self.screen_width // 2 - 15, y_offset))

    def set_friction_map(self, friction_map):
        self.friction_map = friction_map[0]
        # plot with matplotlib
        #plt.plot(self.friction_map)
        #plt.show()

        self._cache_track()

    def close(self):
        pygame.quit()
