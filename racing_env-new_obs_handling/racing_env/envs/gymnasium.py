from typing import Optional
import gymnasium as gym
from gymnasium.vector import VectorEnv

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from racing_env.envs.simulators.simulator import Simulator
from racing_env.utils.constants import CAR, OBSERVATION, TRACK
from racing_env.utils.defaults import DEFAULTS
from racing_env.utils.graphics import SceneRenderer
from racing_env.utils.utils import make_vec_space, to_numpy

vc_action_space = gym.spaces.Box(
        low=np.array([-1, 0]),
        high=np.array([1] * 2),
        shape=(2,),
        dtype=np.float32,
    )



class SingleTrackVecEnv(gym.Env):
    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        dt: float = 0.05,
        frame_skip: int = 5,
        tracks: list = DEFAULTS.TRACKS,
        render_mode: str = "human",
        action_space: gym.spaces.Space = DEFAULTS.ACTION_SPACE,
        rand_config: Optional[str] = None,
        randomize_if_done: bool = True,
        learned_model_config: Optional[str] = None,
        lidar_ray_file: str = None,
        initializer: str = "zero",
        normalize_observations: bool = False,
        reset_if_off_track: bool = True,
        two_way_tracks: bool = False,
        compile: bool = True,
        device: torch.device = torch.device("cpu"),
        exclude_history: bool = False,
        exclude_friction: bool = False,
        exclude_track_boundaries: bool = False,
        observation_config: str = "basic",
        reward_type : str = "basic", # Available: "progress_and_collision", "basic"
        action_delay_seconds: float = 0.0,
        vehicle_config: str = "xray_ldm",
    ):
        self.device = device
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.frame_skip = frame_skip
        self.normalize_observations = normalize_observations
        self.learned_model_config = learned_model_config

        self.single_action_space = action_space
        self.action_space = make_vec_space(self.single_action_space, num_envs)

        simulator_kwargs = dict(
            vehicle_config=vehicle_config,
            tire_config="pacejka",
            learned_model_config=learned_model_config,
            rand_config=rand_config,
            randomize_if_done=randomize_if_done,
            dt=self.dt,
            frame_skip=self.frame_skip,
            maximum_duration=max_episode_steps * dt,
            integration_method="rk4",
            num_envs=num_envs,
            lidar_ray_file=lidar_ray_file,
            initializer=initializer,
            tracks=tracks,
            device=device,
            reset_if_off_track=reset_if_off_track,
            two_way_tracks=two_way_tracks,
            compile=compile,
            exclude_history=exclude_history,
            exclude_friction=exclude_friction,
            exclude_track_boundaries=exclude_track_boundaries,
            reward_type = reward_type,
            observation_config=observation_config,
            action_delay_seconds=action_delay_seconds,
        )

        self.simulator = Simulator(**simulator_kwargs)

        # create observation space
        self.single_observation_space = self.get_observation_space()
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n=num_envs)

        vehicle_params = self.simulator.get_vehicle_params()
        track_x, track_y, track_width = self.simulator.get_track()
        friction_map = self.simulator.get_friction_map()
        print("Got friction")

        # Convert track to numpy
        track_x = track_x.cpu().detach().numpy()
        track_y = track_y.cpu().detach().numpy()
        track_width = track_width.cpu().detach().numpy()

        #TODO: Add handling of difrent track points config

        self.screen_renderer = SceneRenderer(
            vehicle_params=vehicle_params,
            track_x=track_x,
            track_y=track_y,
            track_width=track_width,
            scale=80 / 1,
            dt=self.dt,
            friction_map=friction_map,
        )

        # Needed for stable baselines
        self.render_mode = [render_mode] * num_envs

    def step(self, actions):
        actions = torch.tensor(actions, device=self.simulator.device)
        state, reward, observation, terminated, truncated, info = self.simulator.forward(
            actions
        )

        reward = to_numpy(reward)
        observation = to_numpy(observation)  # dict {"policy": np.ndarray, "critic": np.ndarray}
        terminated = to_numpy(terminated)
        truncated = to_numpy(truncated)
        info['final_obs'] = to_numpy(info['final_obs'])

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        state, info, observation = self.simulator.reset()

        if self.screen_renderer._rendering_started:
            self.screen_renderer.set_friction_map(self.simulator.get_friction_map())

        observation = to_numpy(observation)  # dict {"policy": np.ndarray, "critic": np.ndarray}
        return observation, info

    def pop_statistics(self):
        return self.simulator.pop_statistics()

    def default_controls(self, horizon=1):
        return self.simulator.default_controls(horizon)

    def start_render(self):
        self.screen_renderer.start_render()

    def render(self, obs=None, idxs=None, trajectories=None, costs=None, mode="human"):
        state = self.simulator.state.clone()
        closest_idx = self.simulator.get_closest_idx()
        lidar_data = None  # TOD FIX obs['lidar'] if obs is not None and 'lidar' in obs else None
        self.screen_renderer.render(
            [0] if idxs is None else idxs, closest_idx, state, trajectories, costs, lidar_data=lidar_data
        )

    def set_friction_curve(self, track_id, friction_spline_x, friction_spline_y):
        self.simulator.set_friction_curve(
            track_id,
            friction_spline_x,
            friction_spline_y,
            )
        if self.screen_renderer._rendering_started:
            self.screen_renderer.set_friction_map(self.simulator.get_friction_map())

    def set_friction_curve_all(self, friction_spline_x, friction_spline_y):
        """
        Set the same friction curve for all environments in the vectorized env.
        """
        self.simulator.set_friction_curve_all(friction_spline_x, friction_spline_y)
        if self.screen_renderer._rendering_started:
            self.screen_renderer.set_friction_map(self.simulator.get_friction_map())

    @property
    def state(self):
        return self.simulator.state

    def close(self):
        self.simulator.close()

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            return getattr(self, attr_name)
        else:
            return getattr(self, attr_name)[indices]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            setattr(self, attr_name, value)
        else:
            getattr(self, attr_name)[indices] = value

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if indices is None:
            return getattr(self, method_name)(*method_args, **method_kwargs)
        else:
            return [
                getattr(self, method_name)(ind, *method_args, **method_kwargs)
                for ind in indices
            ]
    
    def get_observation_space(self):
        return self.simulator.observation_creator.get_observation_space()

    def normalize_obs(self, observations):
        for k, v in self.observation_space.items():
            if isinstance(observations[k], torch.Tensor):
                observations[k] = observations[k].cpu().detach().numpy()
            if isinstance(v, gym.spaces.Box):
                if np.isfinite(v.low).all() and np.isfinite(v.high).all(): 
                    #observations[k] = (observations[k] - v.low) / (v.high - v.low)
                    observations[k] = 2 * (observations[k] - v.low) / (v.high - v.low) - 1.
        return observations


class SingleTrackEnv(SingleTrackVecEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(num_envs=1, *args, **kwargs)


gym.register(
    id="racing_env/SingleTrack-v0",
    entry_point=SingleTrackEnv,
    vector_entry_point=SingleTrackVecEnv,
)