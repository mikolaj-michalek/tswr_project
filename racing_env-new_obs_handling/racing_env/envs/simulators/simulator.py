import os
import torch
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from torch.func import functional_call

from racing_env.envs.initializers.handler import initializer_handler
from racing_env.envs.rewards.progress_and_collisions import ProgressAndCollisionReward
from racing_env.envs.rewards.basic_reward import BasicProgressAndCollisionReward
from racing_env.envs.rewards.double_after_lap import DoubleAfterLapReward
from racing_env.envs.rewards.progress_collisions import ProgressCollisionReward
from racing_env.envs.simulators.basic_simulator import BasicSimulator
from racing_env.envs.simulators.robot_models.learned_model import LearnedModel
from racing_env.utils.constants import CAR, OBSERVATION, STATE, TRACK
from racing_env.utils.controls import nominal_controls_along_track
from racing_env.utils.delay_fifo import DelayBufferFIFO
from racing_env.utils.history_buffer import HistoryBuffer
from racing_env.utils.state_wrapper import STATE_DEF_LIST
from racing_env.utils.track_geometry import compute_position_on_track, get_ego_transformed_track_boundaries, get_friction_at_positions
from racing_env.utils.tracks import load_tracks
from racing_env.utils.param_randomizer import ParamRandomizer
from racing_env.envs.sensors.lidar import LidarSensor
from racing_env.envs.obs.obs_creator import ObservationCreator

# ML
from racing_env.envs.rewards.ml_drift360_reward import Drift360Reward

 
class Simulator(BasicSimulator):
    @torch.no_grad()
    def __init__(
        self,
        vehicle_config: str,
        tire_config: str,
        learned_model_config: str,
        dt: float,
        maximum_duration: float,
        integration_method: str,
        num_envs: int,
        tracks: list,
        device: torch.device,
        lidar_ray_file: str = None,
        reward_type : str = "basic",  # Available: "progress_and_collision", "basic", "double_after_lap"
        observation_config: str = "basic",
        rand_config: str = None,
        randomize_if_done: bool = True,
        initializer: str = "zero",
        reset_if_off_track: bool = True,
        two_way_tracks: bool = False,
        frame_skip: int = 1,
        compile: bool = True,
        exclude_history: bool = False,
        exclude_friction: bool = False,
        exclude_track_boundaries: bool = False,
        action_delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            dt=dt,
            frame_skip=frame_skip,
            maximum_duration=maximum_duration,
            integration_method=integration_method,
            num_envs=num_envs,
            randomize=rand_config is not None,
            device=device,
        )
        logging.info("Initializing simulator")

        # Action delay: seconds -> steps, two FIFO buffers for steering and omega_dot
        self.action_delay_seconds = action_delay_seconds
        delay_steps = max(0, int(action_delay_seconds / self.dt))
        if delay_steps > 0:
            self._steering_delay = DelayBufferFIFO(delay_steps, num_envs, device=device)
            self._omega_dot_delay = DelayBufferFIFO(delay_steps, num_envs, device=device)

        # Config
        self.compile = compile
        torch.set_default_device(self.device)
        self.reset_if_off_track = reset_if_off_track
        self.two_way_tracks = two_way_tracks
        self.randomize_if_done = randomize_if_done

        self.observation_creator = ObservationCreator(
            file_name=observation_config,
            num_env=num_envs,
            normalize=True,
        )

        self.lidar = None
        if lidar_ray_file:
            # Construct full path if it's just a filename, or use as is
            # Assuming files are in tracks/track_name/rays... 
            # You might need to adjust path logic depending on where you save the npz
            self.lidar = LidarSensor(
                ray_file_path=lidar_ray_file,
                num_envs=num_envs,
                device=device
            )
        
        
        self.single_track_model = LearnedModel(learned_model_config)

        self.reward = None

        if reward_type == "progress_and_collision":
            self.reward = ProgressAndCollisionReward(self)
        elif reward_type == "progress_collision":
            self.reward = ProgressCollisionReward(self)
        elif reward_type == "basic":
            self.reward = BasicProgressAndCollisionReward(self)
        elif reward_type == "double_after_lap":
            self.reward = DoubleAfterLapReward(self)
        # ML
        elif reward_type == "drift360":
            self.reward = Drift360Reward(self)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

        self.exclude_history = exclude_history
        self.exclude_friction = exclude_friction
        self.exclude_track_boundaries = exclude_track_boundaries

        # Compile methods
        if self.compile:
            self.single_track_model = torch.compile(
                self.single_track_model,
                mode='max-autotune-no-cudagraphs',
            )
            self.get_ego_transformed_track_boundaries = torch.compile(
                self.get_ego_transformed_track_boundaries,
                mode='max-autotune-no-cudagraphs',
            )
            self.get_friction_map = torch.compile(
                self.get_friction_map,
                mode='max-autotune-no-cudagraphs',
            )
            self.get_curvature_map = torch.compile(
                self.get_curvature_map,
                mode='max-autotune-no-cudagraphs',
            )
            self.compute_position_on_track_ = torch.compile(
                self.compute_position_on_track,
                mode='max-autotune-no-cudagraphs',
            )
            print("Compiled single track model and utility methods")

            self.forward = torch.compile(
                self.forward,
                mode='max-autotune-no-cudagraphs',
            )
            self._calculate_observation = torch.compile(
                self._calculate_observation,
                mode='max-autotune-no-cudagraphs',
            )
            logging.info("Methods compiled")

        # State [x, y, yaw, v_x, v_y, r, omega_wheels, omega_wheels_ref, delta, friction, delta_ref, omega_dot]
        self.state = torch.zeros((num_envs, STATE.SIZE), dtype=torch.float32, device=self.device)
        self.ax = torch.zeros((num_envs), dtype=torch.float32, device=self.device)
        self.ay = torch.zeros((num_envs), dtype=torch.float32, device=self.device)
        self.ep_progress = torch.zeros((num_envs), dtype=torch.float32, device=self.device)
        self.vehicle_dynamics_metrics: dict[str, torch.Tensor] = {}
        self.ep_duration = torch.zeros((num_envs), dtype=torch.float32, device=self.device)
        self.state_names = STATE_DEF_LIST
        self.control_names = ["delta", "omega_dot"]

        self._load_params(
            vehicle_config=vehicle_config,
            rand_config=rand_config,
        )

        self._load_tracks(tracks)

        # Pre-computed index buffers — avoids repeated torch.arange allocation every step
        self._batch_idx = torch.arange(self.num_envs, device=self.device)
        self._batch_idx_2d = self._batch_idx.unsqueeze(1)
        self._track_range = torch.arange(TRACK.MAX_SIZE, device=self.device)
        self._track_range_2d = self._track_range.unsqueeze(0)

        self.state_initializer = initializer_handler(initializer, self)
        self._initialize_history()

        if self.randomize:
            self.param_randomizer = ParamRandomizer(
                rand_params=self.rand_params,
                device=self.device,
                model_params_orig=self.model_params,
                track_size=self.track_size,
                num_envs=self.num_envs,
            )
            self._randomize_params()
        else:
            self.param_randomizer = None

        self._reset_state()
        logging.info("Simulator initialized")

    @torch.no_grad()
    def forward(self, u):
        """Simulates one step of the vehicle dynamics given control inputs.

        Args:
            u (torch.tensor): Control inputs tensor of shape [batch_size, 2] containing:
                - u[:,0]: Normalized steering angle in range [-1,1]
                - u[:,1]: Normalized motor current in range [-1,1]

        Returns:
            tuple: (state, reward, observation, terminated, truncated, info)
                observation is dict {"policy": Tensor, "critic": Tensor}
        """
        # clip steering and current control between -1 and 1
        u[:, 0] = torch.clamp(u[:, 0], -1, 1)
        u[:, 1] = torch.clamp(u[:, 1], -1, 1)

        # Apply action delay if configured
        if hasattr(self, "_steering_delay"):
            u = torch.stack([
                self._steering_delay(u[:, 0]),
                self._omega_dot_delay(u[:, 1]),
            ], dim=1)

        # Scale to physical units
        steering = u[:, 0] * CAR.MAX_STEERING
        current = u[:, 1] * CAR.MAX_OMEGA_DOT

        u = torch.stack([steering, current], dim=1)

        front_friction, rear_friction = self.get_front_tire_friction(), self.get_rear_tire_friction()

        for i in range(self.frame_skip):
            self._step(u, front_friction, rear_friction)


        observation = self._calculate_observation()
        reward = self.reward()

        # Populate replay buffer with states visited in this step.
        # Only non-off-track environments are recorded so the buffer contains
        # clean, drivable states that are worth revisiting during training.
        if hasattr(self.state_initializer, "record_states"):
            valid = ~self.off_track
            if valid.any():
                self.state_initializer.record_states(
                    self.state[valid],
                    self.closest_idx[valid],
                    self.last_s[valid],
                )

        self.ep_duration += self.dt
        self.ep_progress += self.progress
        self.metrics.step(reward, self.progress, self.off_track, self.slightly_off_track, self.dt)

        terminated = self.off_track if self.reset_if_off_track else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.ep_duration > self.maximum_duration
        #truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #done = terminated 

        info = {"final_obs": [None] * self.num_envs}
        if terminated.any():
            self.metrics.count_resets(terminated.sum().item())
            self.off_track = terminated
            observation = self._reset_state(mask=self.off_track)
            if self.randomize and self.randomize_if_done:
                self._randomize_params(mask=self.off_track)

        return self.state, reward, observation, terminated, truncated, info

    def _step(self, u, front_friction, rear_friction):
        """
        u: [delta, omega_dot]
        front_friction, rear_friction: [batch_size]
        """
        self.state[:, 9] = front_friction
        self.state[:, 10] = rear_friction
        self.state[:, 11] = u[:, 0]  # delta_ref [rad]
        self.state[:, 12] = u[:, 1]  # omega_dot - wheel speed derivative [m/s] 

        # save previous velocities for acceleration calculation
        vx_prev = self.state[:, 3].clone()
        vy_prev = self.state[:, 4].clone()

        def _model_forward(t, x):
            return functional_call(self.single_track_model, self.model_params, (t, x))

        self.state = self._integrate(_model_forward, self.state, self.timestep)

        self.state[:, 3] = torch.clamp(self.state[:, 3], CAR.MIN_VELOCITY, CAR.MAX_VELOCITY)
        self.state[:, 6] = torch.clamp(self.state[:, 6], CAR.MIN_VELOCITY+self.eps, 2. * CAR.MAX_VELOCITY)
        self.state[:, 7] = torch.clamp(self.state[:, 7], CAR.MIN_VELOCITY+self.eps, CAR.MAX_OMEGA_REF)
        self.state[:, 2] = torch.atan2(
            torch.sin(self.state[:, 2]), torch.cos(self.state[:, 2])
        )

        # calculate accelerations of car including steering effects
        # a_x = dv_x/dt - v_y * r; a_y = dv_y/dt + v_x * r
        self.ax = (self.state[:, 3] - vx_prev) / self.dt - self.state[:, 4] * self.state[:, 5]
        self.ay = (self.state[:, 4] - vy_prev) / self.dt + self.state[:, 3] * self.state[:, 5]

        # Store vehicle dynamics metrics during integration
        v_x, v_y = self.state[:, 3], self.state[:, 4]
        self.vehicle_dynamics_metrics = {
            "beta_dyn": torch.atan2(v_y, v_x),
            **(
                self.single_track_model.last_tire_dynamics
                if hasattr(self.single_track_model, "last_tire_dynamics")
                else {}
            ),
        }

    # TODO make the method use indices for which the computations should be done
    @torch.no_grad()
    def _calculate_observation(self):
        # Get state components
        pos = self.state[:, :2]  # [batch_size, 2]
        yaw = self.state[:, 2]   # [batch_size]

        self.closest_idx, p1_idx, p2_idx, t, signed_dist_to_centerline, signed_dist_to_edge, min_dist_to_edge, self.tire_dists_to_edge = \
            self.compute_position_on_track(pos=pos, yaw=yaw)
        
        # Interpolate s and heading
        batch_idx = self._batch_idx
        now_s = torch.lerp(self.track_s[batch_idx, p1_idx], self.track_s[batch_idx, p2_idx], t)
        heading = torch.lerp(self.track_heading[batch_idx, p1_idx], self.track_heading[batch_idx, p2_idx], t)
        local_curvature = torch.lerp(self.track_curvature[batch_idx, p1_idx], self.track_curvature[batch_idx, p2_idx], t)
        # Front/rear tire positions for friction sampling
        lf = self._get_lf()
        lr = self._get_lr()
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        front_pos = pos + torch.stack([lf * cos_yaw, lf * sin_yaw], dim=-1)
        rear_pos = pos - torch.stack([lr * cos_yaw, lr * sin_yaw], dim=-1)
        local_front_friction = get_friction_at_positions(
            front_pos, self.track_x, self.track_y, self.track_size, self.friction,
            center_idx=self.closest_idx,
        )
        local_rear_friction = get_friction_at_positions(
            rear_pos, self.track_x, self.track_y, self.track_size, self.friction,
            center_idx=self.closest_idx,
        )

        self.progress = now_s - self.last_s
        # Handle track loop (if progress is big and negative car looped around)
        self.progress = torch.where(
            torch.abs(self.progress) > 0.95 * self.track_lengths,
            self.progress % self.track_lengths,
            self.progress
        )
        self.last_s = now_s

        # Calculate heading difference
        heading_diff = yaw - heading
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        self.heading_diff = heading_diff

        # Calculate velocity in Frenet frame
        v_x = self.state[:, 3]
        v_y = self.state[:, 4]
        self.v_f = ((v_x * torch.cos(heading_diff) - v_y * torch.sin(heading_diff)) 
               / (1 - signed_dist_to_centerline * local_curvature))

        # Ensure vehicle_dynamics_metrics populated (needed on initial reset before first _step)
        if not self.vehicle_dynamics_metrics or "Fx_f" not in self.vehicle_dynamics_metrics:
            t = torch.zeros(1, dtype=torch.float32, device=self.device)
            _ = functional_call(self.single_track_model, self.model_params, (t, self.state))
        self.vehicle_dynamics_metrics = {
            "beta_dyn": torch.atan2(v_y, v_x),
            "v_f": self.v_f,
            **(getattr(self.single_track_model, "last_tire_dynamics", {})),
        }

        # Calculate dynamical and kinematic slip angles
        self.beta_dyn = torch.atan2(v_y, v_x)
        self.beta_kin = torch.atan(
            torch.tan(self.state[:, 8]) * self._get_lr() / (self._get_lf() + self._get_lr())
        )
        
        # Check if off track
        self.signed_dist_to_edge = signed_dist_to_edge
        self.off_track = min_dist_to_edge < -TRACK.OFF_TRACK_DISTANCE
        # self.off_track = signed_dist_to_edge < 0
        self.slightly_off_track = torch.logical_and(min_dist_to_edge < 0.,
                                                    min_dist_to_edge > -TRACK.OFF_TRACK_DISTANCE)
        self.signed_dist_to_centerline = signed_dist_to_centerline

        self.steps_no_off_track = torch.where(self.off_track, torch.zeros(1, device=self.device), self.steps_no_off_track + 1)

        import math
        s_normalized = now_s / self.track_lengths  # [num_envs], in [0, 1]

        laps_finished = (
            self.ep_progress / self.track_lengths
        ).long().clamp(min=0).float()

        env_metrics = {
            "laps_finished": laps_finished,
            "heading_diff": heading_diff,
            "front_tire_friction": local_front_friction,
            "rear_tire_friction": local_rear_friction,
            # Normalised progress in [0, 1]: the source for transform observations.
            # YAML entries with `transform: sin/cos` look this up by name.
            "progress": s_normalized,
            "signed_dist_to_edge": signed_dist_to_edge,
            "signed_dist_to_centerline": signed_dist_to_centerline,
            "min_dist_to_edge": min_dist_to_edge,
            "tire_dists_to_edge": self.tire_dists_to_edge,  # [batch, 4] FL FR RL RR
            "ax": self.ax,
            "ay": self.ay,
            **self.vehicle_dynamics_metrics,
        }

        # Record current vehicle state + controls into the history buffer.
        # Stored features (8-dim, matching obs_dim in _initialize_history):
        #   v_x, v_y, r, delta, delta_ref, omega_wheels, omega_wheels_ref, omega_wheels_ref_dot
        raw_hist = torch.stack([
            self.state[:, 3],   # v_x
            self.state[:, 4],   # v_y
            self.state[:, 5],   # r
            self.state[:, 8],   # delta
            self.state[:, 11],  # delta_ref
            self.state[:, 6],   # omega_wheels
            self.state[:, 7],   # omega_wheels_ref
            self.state[:, 12],  # omega_wheels_ref_dot
        ], dim=-1)
        self.vehicle_state_history.add(raw_hist)

        history = self.vehicle_state_history.get_history() if not self.exclude_history else None
        friction_map = self.get_friction_map() if not self.exclude_friction else None
        curvature_map = self.get_curvature_map()
        track_boundaries = self.get_ego_transformed_track_boundaries() if not self.exclude_track_boundaries \
                                                                        else [None] * self.state.shape[0]
        
        # Calcuating lidar observation
        lidar_obs = None
        if self.lidar is not None:
            # x, y, yaw are already available in self.state
            lidar_obs = self.lidar.get_observation(
                x=self.state[:, 0],
                y=self.state[:, 1],
                yaw=self.state[:, 2]
            )

        policy_obs, critic_obs = self.observation_creator.create_obs(
            state=self.state,
            env_metrics=env_metrics,
            friction_map=friction_map,
            curvature_map=curvature_map,
            history=history,
        )

        return {"policy": policy_obs, "critic": critic_obs}

    def _reset_history(self, mask=None):
        if mask is None:
            mask = torch.ones(self.num_envs, device=self.device).bool()
        self.vehicle_state_history.reset(mask)
        if hasattr(self, "_steering_delay"):
            self._steering_delay.reset(mask)
            self._omega_dot_delay.reset(mask)
        if self.observation_creator.obs_randomizer is not None:
            self.observation_creator.obs_randomizer.reset_biases(mask)
        self.observation_creator.reset_source_biases(mask)

    def default_controls(self, horizon: int = 1):
        return nominal_controls_along_track(s=self.last_s, vx=self.state[:, 3],
            track_s=self.track_s, track_curvature=self.track_curvature,
            car_length=self._get_lf() + self._get_lr(),
            dt=self.dt, horizon=horizon)

    def _load_params(self, vehicle_config, rand_config):
        # Build per-env model_params dict from the model's own buffers/parameters.
        self.model_params = self.single_track_model.build_params_dict(self.num_envs)
        # Move to correct device
        self.model_params = {k: v.to(self.device) for k, v in self.model_params.items()}

        if rand_config is not None:
            config_path = os.path.join(os.path.dirname(__file__), "config")
            with open(os.path.join(config_path, f"randomization/{rand_config}.yaml")) as f:
                rand_params_config = yaml.load(f, Loader=yaml.FullLoader)
                self.rand_params = rand_params_config["Params"]
                print(f"Loaded randomization config: {rand_config}")

    def _randomize_params(self, mask=None):
        """Randomize parameters using the ParamRandomizer class."""
        if mask is None:
            indices = None  # randomize all
        else:
            indices = mask.nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                return  # nothing to do
        new_model_params, new_friction = self.param_randomizer.randomize(indices=indices)
        if indices is None:
            for k, v in self.model_params.items():
                self.model_params[k] = new_model_params[k]
            self.friction[:, :new_friction.shape[1]] = new_friction
        else:
            for k, v in self.model_params.items():
                self.model_params[k][indices] = new_model_params[k]
            self.friction[indices, :new_friction.shape[1]] = new_friction

    def get_front_tire_friction(self) -> torch.Tensor:
        pos = self.state[:, :2]
        yaw = self.state[:, 2]
        lf = self._get_lf()
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        front_pos = pos + torch.stack([lf * cos_yaw, lf * sin_yaw], dim=-1)
        return get_friction_at_positions(
            front_pos, self.track_x, self.track_y, self.track_size, self.friction,
            center_idx=self.closest_idx,
        )

    def get_rear_tire_friction(self) -> torch.Tensor:
        pos = self.state[:, :2]
        yaw = self.state[:, 2]
        lr = self._get_lr()
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        rear_pos = pos - torch.stack([lr * cos_yaw, lr * sin_yaw], dim=-1)
        return get_friction_at_positions(
            rear_pos, self.track_x, self.track_y, self.track_size, self.friction,
            center_idx=self.closest_idx,
        )
    
    def set_friction_curve(self, env_idx, x_points, y_points):
        """
        Set the friction curve for a specific environment.

        Args:
            env_idx (int): Index of the environment.
            x_points (list): List of x-coordinates for the friction curve noralize to 0, 1.
            y_points (list): List of y-coordinates for the friction curve.
        """
        track_size = self.track_size[env_idx].cpu().numpy()
        x_points = np.array(x_points, dtype=np.float32)
        x_points = np.clip(x_points, 0, 1) * (track_size - 1) 
        x_values = np.linspace(0, track_size - 1, track_size)

        # print(f"Setting friction curve for env {env_idx} with x_points: {x_points} and y_points: {y_points}")

        # Interpolate between control points
        interpolated_values = np.interp(x_values, x_points, y_points)

        # Smooth the resulting profile
        smoothed_values = gaussian_filter1d(interpolated_values, 5, mode="wrap")
        smoothed_values = torch.tensor(smoothed_values, device=self.device)

        # Set the friction values
        self.friction[env_idx, :track_size] = smoothed_values

        # plot 
        #plt.plot(x_values, smoothed_values.cpu().numpy(), label=f"env {env_idx}")

    def set_friction_curve_all(self, x_points, y_points):
        """
        Set the same friction curve for all environments (vectorized over envs).

        Args:
            x_points (list): List of x-coordinates for the friction curve normalized to [0, 1].
            y_points (list): List of y-coordinates for the friction curve.
        """
        # Assume all envs use the same discretization for the track length
        track_size = int(self.track_size[0].cpu().item())

        x_points_arr = np.array(x_points, dtype=np.float32)
        x_points_arr = np.clip(x_points_arr, 0.0, 1.0) * (track_size - 1)
        x_values = np.linspace(0, track_size - 1, track_size, dtype=np.float32)

        interpolated_values = np.interp(x_values, x_points_arr, y_points)
        smoothed_values = gaussian_filter1d(interpolated_values, 5, mode="wrap")
        smoothed_values_t = torch.tensor(smoothed_values, device=self.device)

        # Broadcast the same profile to all environments in one shot
        self.friction[:, :track_size] = smoothed_values_t.unsqueeze(0)

    def _load_tracks(self, tracks: list) -> None:
        self.numbers_of_tracks, self.track_lengths, self.track_size, self.track_x, self.track_y, \
        self.track_heading, self.track_width, self.track_curvature, self.track_s, self.tracks_idx \
        = load_tracks(tracks, self.num_envs, TRACK.MAX_SIZE, self.two_way_tracks, device=self.device)

        # Create friction tensor
        self.friction = torch.ones_like(self.track_x) * TRACK.DEFAULT_FRICTION

    def _reset_state(self, mask=None, start_at_zero: bool = False):
        self.steps_no_off_track = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        if mask is None:
            self.ax = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
            self.ay = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
            self.ep_progress = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
            self.ep_duration = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        else:
            self.ax = torch.where(mask, 0.0, self.ax)
            self.ay = torch.where(mask, 0.0, self.ay)
            self.ep_progress = torch.where(mask, 0.0, self.ep_progress)
            self.ep_duration = torch.where(mask, 0.0, self.ep_duration)
        return super()._reset_state(mask=mask, start_at_zero=start_at_zero)

    def _initialize_history(self):
        self.vehicle_state_history = HistoryBuffer(
            OBSERVATION.HISTORY_SIZE, self.num_envs, obs_dim=8, device=self.device  # TODO change 8 to value from obs_dim
            )

    def get_ego_transformed_track_boundaries(self):
        """
        Transforms the next N points on the track to the ego vehicle's frame of reference.
        
        Returns:
            torch.Tensor: Points in ego frame, shape [batch_size, n_points, 4]
            Each point contains [x_left, y_left, x_right, y_right]
        """
        return get_ego_transformed_track_boundaries(x_ego=self.state[:, 0], y_ego=self.state[:, 1],
            yaw_ego=self.state[:, 2], s=self.last_s, track_s=self.track_s, track_x=self.track_x, track_y=self.track_y,
            track_width=self.track_width, track_heading=self.track_heading,
            foresight_spacing=OBSERVATION.FORESIGHT_SPACING,
            foresight_size=OBSERVATION.FORESIGHT_SIZE
        )

    def compute_position_on_track(self, pos, yaw, vehicle_params=None):
        """Compute the closest point on the track.

        ``vehicle_params`` is assembled from the named scalar buffers
        ``lf``/``lr`` (or ``L``/``lr``) in ``model_params`` unless provided.
        """
        if vehicle_params is None:
            vehicle_params = self._build_vehicle_params_for_geometry()
        return compute_position_on_track(
            pos=pos,
            yaw=yaw,
            vehicle_params=vehicle_params,
            track_x=self.track_x,
            track_y=self.track_y,
            track_width=self.track_width,
            track_size=self.track_size
        )
    
    def get_friction_map(self):
        """
        Get all friction values starting from closest_idx, wrapping around the track.
        """
        # Create indices
        indices = (self.closest_idx.unsqueeze(1) + self._track_range_2d) % self.track_size.unsqueeze(1)

        # Get friction values
        friction = self.friction[self._batch_idx_2d, indices]

        # Mask out values beyond actual track size
        mask = self._track_range_2d < self.track_size.unsqueeze(1)
        friction = friction * mask

        return friction

    def get_curvature_map(self) -> torch.Tensor:
        """Curvature values from closest_idx forwards, wrapping around track."""
        indices = (self.closest_idx.unsqueeze(1) + self._track_range_2d) % self.track_size.unsqueeze(1)
        curvature = self.track_curvature[self._batch_idx_2d, indices]
        mask = self._track_range_2d < self.track_size.unsqueeze(1)
        return curvature * mask

    def set_friction(self, friction):
        """
        Args:
            friction (torch.Tensor): New friction value [num_envs, track_size]
        """
        self.base_friction = torch.full((self.num_envs, self.track_size[0]), friction, device=self.device)

    def get_track(self):
        """
        Returns:
            torch.Tensor: x coordinates of the track [track_size]
            torch.Tensor: y coordinates of the track [track_size]
            torch.Tensor: width of the track [track_size]
        """
        # if more than one track log error 
        if len(self.tracks_idx.keys()) > 1:
            logging.error("More than one track loaded, returning only first track")

        # return the track without padding
        size = self.track_size[0]
        return self.track_x[0, :size], self.track_y[0, :size], self.track_width[0, :size]
    
    def get_closest_idx(self):
        """
        Returns:
            torch.Tensor: Index of the closest point on the track [batch_size]
        """
        return self.closest_idx
    
    def get_heading_diff(self):
        """
        Returns:
            float: Heading difference between the vehicle and the track
        """
        return self.heading_diff.item()

    def get_signed_dist_to_centerline(self):
        """
        Returns:
            torch.Tensor: Signed distance to the centerline [batch_size]
        """
        return self.signed_dist_to_centerline
    
    def get_vehicle_params(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Vehicle parameters [batch_size, num_vehicle_params]
        """
        return self._build_vehicle_params_for_geometry()

    # ------------------------------------------------------------------
    # Geometry helpers — read named scalar buffers from model_params
    # ------------------------------------------------------------------

    def _get_L(self) -> torch.Tensor:
        """Wheelbase [num_envs]."""
        return self.model_params["learned_model.vehicle_parameters.L"]

    def _get_lf(self) -> torch.Tensor:
        """Front semi-wheelbase [num_envs]."""
        # lf = L - lr  (standard naming)
        L = self._get_L()
        lr = self._get_lr()
        return (L - lr)

    def _get_lr(self) -> torch.Tensor:
        """Rear semi-wheelbase [num_envs]."""
        return self.model_params["learned_model.vehicle_parameters.lr"]

    def _build_vehicle_params_for_geometry(self) -> torch.Tensor:
        """Build a ``[num_envs, 16]`` vehicle-params tensor compatible with
        ``VehicleParameters`` / ``track_geometry.compute_position_on_track``.

        Only columns 3 (L = lf+lr) and 4 (lr) are used by the geometry code.
        All other columns are filled with zeros.
        """
        n = self.num_envs
        lr = self._get_lr()
        L = self._get_L()
        p = torch.zeros(n, 16, device=self.device)
        p[:, 3] = L
        p[:, 4] = lr
        return p