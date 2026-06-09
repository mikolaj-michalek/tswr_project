import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
import gymnasium as gym
import torch

from racing_env.utils.state_wrapper import StateWrapper
from racing_env.utils.obs_randomizer import ObsRandomizer


# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------
# Transforms are applied to a normalised source signal in [0, 1].
# Convention: "sin" means sin(x * 2π), "sin_x2" means sin(2 * x * 2π), etc.
# ---------------------------------------------------------------------------
_TWO_PI = 2.0 * math.pi

_TRANSFORMS: dict[str, callable] = {
    "sin":    lambda x: torch.sin(x * _TWO_PI),
    "cos":    lambda x: torch.cos(x * _TWO_PI),
    "sin_x2": lambda x: torch.sin(2.0 * x * _TWO_PI),
    "cos_x2": lambda x: torch.cos(2.0 * x * _TWO_PI),
    "sin_x4": lambda x: torch.sin(4.0 * x * _TWO_PI),
    "cos_x4": lambda x: torch.cos(4.0 * x * _TWO_PI),
}


# ---------------------------------------------------------------------------
# ObservationInfo
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ObservationInfo:
    name: str
    dim: int
    min_val: float
    max_val: float
    std: float = 0.0        # noise std applied to the source (before transform, or to the raw obs)
    bias_std: float = 0.0   # per-env constant offset std; resampled once per episode
    privileged: bool = False
    iterable: bool = field(init=False)
    length: int | None = None
    sample_rate: int | None = None
    feature_dim: int | None = None  # obs_history: feature dim per time-step
    transform: str | None = None

    def __post_init__(self):
        self.iterable = self.length is not None
        if self.transform is not None and self.transform not in _TRANSFORMS:
            raise ValueError(
                f"ObservationInfo '{self.name}': unknown transform '{self.transform}'. "
                f"Available: {sorted(_TRANSFORMS)}"
            )


# ---------------------------------------------------------------------------
# ObservationCreator
# ---------------------------------------------------------------------------

class ObservationCreator:
    def __init__(
        self,
        file_name: str,
        num_env: int,
        randomize: bool = False,
        normalize: bool = True,
    ):
        self.observation_list: list[ObservationInfo] = []
        self.num_env = num_env
        self.randomize = randomize
        self.normalize = normalize
        self.obs_randomizer: ObsRandomizer | None = None

        # Separate tracking for policy (non-privileged) and critic (all) obs
        self.policy_obs_dim = 0
        self.critic_obs_dim = 0
        self._policy_min: list[float] = []
        self._policy_max: list[float] = []
        self._critic_min: list[float] = []
        self._critic_max: list[float] = []
        self._policy_std: list[float] = []
        self.policy_obs_names: list[str] = []
        self.critic_obs_names: list[str] = []

        self._load_from_file(file_name)

        # ---------------------------------------------------------------
        # Column-level noise (non-transform obs)
        # ---------------------------------------------------------------
        if randomize and any(
            getattr(o, "transform", None) is None
            and ((o.std or 0.0) > 0.0 or (o.bias_std or 0.0) > 0.0)
            for o in self.observation_list
        ):
            self.obs_randomizer = ObsRandomizer(
                observation_list=self.observation_list,
                num_envs=num_env,
            )

        # ---------------------------------------------------------------
        # Source-level noise (transform-derived obs)
        # Collect unique source names and their noise params.  When the same
        # source appears with multiple transforms (sin + cos), we take the
        # maximum noise across all of them (they should agree; max is safe).
        # ---------------------------------------------------------------
        _src_noise: dict[str, list[float]] = {}  # name → [std, bias_std]
        for obs in self.observation_list:
            if obs.transform is None:
                continue
            std = obs.std or 0.0
            bias_std = obs.bias_std or 0.0
            if obs.name in _src_noise:
                prev = _src_noise[obs.name]
                _src_noise[obs.name] = [max(prev[0], std), max(prev[1], bias_std)]
            else:
                _src_noise[obs.name] = [std, bias_std]

        self._source_names: list[str] = list(_src_noise.keys())  # stable order
        self._source_idx: dict[str, int] = {n: i for i, n in enumerate(self._source_names)}

        if self._source_names:
            stds = [_src_noise[n][0] for n in self._source_names]
            bias_stds = [_src_noise[n][1] for n in self._source_names]
            # [1, num_sources] – broadcast over num_envs in the hot path
            self._source_std_t = torch.tensor(stds, dtype=torch.float32).unsqueeze(0)
            self._source_bias_std_t = torch.tensor(bias_stds, dtype=torch.float32).unsqueeze(0)
            self._has_source_step_noise = any(s > 0.0 for s in stds)
            self._has_source_bias = any(b > 0.0 for b in bias_stds)
            # [num_envs, num_sources] – per-episode bias, initialised now
            if self._has_source_bias:
                self._source_bias_t = (
                    torch.randn(num_env, len(self._source_names))
                    * self._source_bias_std_t
                )
            else:
                self._source_bias_t = torch.zeros(num_env, len(self._source_names))
        else:
            self._has_source_step_noise = False
            self._has_source_bias = False
            self._source_bias_t = None
            self._source_std_t = None
            self._source_bias_std_t = None

        # ---------------------------------------------------------------
        # Pre-compute tensors for fast GPU normalization
        # ---------------------------------------------------------------
        self.policy_min_t = torch.tensor(self._policy_min, dtype=torch.float32)
        self.policy_max_t = torch.tensor(self._policy_max, dtype=torch.float32)
        self.critic_min_t = torch.tensor(self._critic_min, dtype=torch.float32)
        self.critic_max_t = torch.tensor(self._critic_max, dtype=torch.float32)
        self.policy_std_t = torch.tensor(self._policy_std, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Episode-reset hook
    # ------------------------------------------------------------------

    def reset_source_biases(self, mask: torch.Tensor | None = None) -> None:
        """Resample per-env source biases for the selected environments.

        Uses ``torch.where`` to stay entirely on GPU with no sync.

        Args:
            mask: Boolean ``[num_env]`` tensor.  ``None`` resets all envs.
        """
        if not self._has_source_bias:
            return
        dev = self._source_bias_t.device
        if mask is None:
            mask = torch.ones(self.num_env, dtype=torch.bool, device=dev)
        elif mask.device != dev:
            mask = mask.to(dev)

        new_biases = (
            torch.randn(self.num_env, len(self._source_names), device=dev)
            * self._source_bias_std_t
        )
        self._source_bias_t = torch.where(mask.unsqueeze(-1), new_biases, self._source_bias_t)

    # ------------------------------------------------------------------
    # Observation table management
    # ------------------------------------------------------------------

    def _add_observation(self, obs: ObservationInfo) -> None:
        self.observation_list.append(obs)

        # Every obs goes into critic
        self.critic_obs_dim += obs.dim
        self._critic_min.extend([obs.min_val] * obs.dim)
        self._critic_max.extend([obs.max_val] * obs.dim)
        self.critic_obs_names.extend([obs.name] * obs.dim)

        # Non-privileged obs also go into policy
        if not obs.privileged:
            self.policy_obs_dim += obs.dim
            self._policy_min.extend([obs.min_val] * obs.dim)
            self._policy_max.extend([obs.max_val] * obs.dim)
            # Transform-derived obs contribute no column-level noise entry
            # (their noise is managed at source level).
            col_std = 0.0 if obs.transform is not None else (obs.std or 0.0)
            self._policy_std.extend([col_std] * obs.dim)
            self.policy_obs_names.extend([obs.name] * obs.dim)

    def _load_from_file(self, name: str) -> None:
        path = os.path.join(os.path.dirname(__file__), "config", f"{name}.yaml")
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        for ob in data:
            self._add_observation(ObservationInfo(
                name=ob["name"],
                dim=ob["dim"],
                min_val=ob["min_val"],
                max_val=ob["max_val"],
                std=ob.get("std", 0.0),
                bias_std=ob.get("bias_std", 0.0),
                privileged=ob.get("privileged", False),
                length=ob.get("length"),
                sample_rate=ob.get("sample_rate"),
                feature_dim=ob.get("feature_dim"),
                transform=ob.get("transform"),
            ))

        print(f"Policy obs ({self.policy_obs_dim}D): {self.policy_obs_names}")
        print(f"Critic obs ({self.critic_obs_dim}D): {self.critic_obs_names}")

    # ------------------------------------------------------------------ #
    #  Iterable obs (e.g. friction map sampled along the track)
    # ------------------------------------------------------------------ #
    def _create_iterable_obs(self, source_tensor: torch.Tensor, obs_info: ObservationInfo) -> torch.Tensor:
        if source_tensor.dim() != 2:
            raise ValueError(
                f"Iterable observation '{obs_info.name}' expects a 2D tensor "
                f"[num_envs, N], got shape {tuple(source_tensor.shape)}"
            )

        num_envs, total_length = source_tensor.shape
        length = obs_info.length if obs_info.length is not None else total_length
        if length > total_length:
            raise ValueError(
                f"Iterable observation '{obs_info.name}' requested length {length}, "
                f"but source tensor only has length {total_length}."
            )

        sample_rate = obs_info.sample_rate if obs_info.sample_rate is not None else 1
        if sample_rate <= 0:
            raise ValueError(
                f"Iterable observation '{obs_info.name}' has invalid sample_rate "
                f"{sample_rate}, must be > 0."
            )

        values = source_tensor[:, :length:sample_rate]
        if values.shape[1] != obs_info.dim:
            raise ValueError(
                f"Iterable observation '{obs_info.name}' produced {values.shape[1]} "
                f"samples, but dim={obs_info.dim} in YAML. "
                f"Ensure dim == ceil(length / sample_rate)."
            )
        return values

    # ------------------------------------------------------------------ #
    #  History obs (obs_history): N past steps of vehicle state/controls
    # ------------------------------------------------------------------ #
    def _create_history_obs(
        self,
        history_tensor: torch.Tensor,
        obs_info: ObservationInfo,
    ) -> torch.Tensor:
        """Slice and flatten a history buffer.

        Args:
            history_tensor: ``[num_envs, history_length, feature_dim]``
                newest entry at index 0.
            obs_info: configuration entry with optional ``length``,
                ``sample_rate``, and ``feature_dim`` fields.

        Returns:
            ``[num_envs, n_steps * feature_dim]`` tensor.
        """
        if history_tensor.dim() != 3:
            raise ValueError(
                f"Observation 'obs_history' expects a 3-D tensor "
                f"[num_envs, H, D], got shape {tuple(history_tensor.shape)}"
            )

        num_envs, history_length, feature_dim = history_tensor.shape
        length = obs_info.length if obs_info.length is not None else history_length
        sample_rate = obs_info.sample_rate if obs_info.sample_rate is not None else 1

        if length > history_length:
            raise ValueError(
                f"obs_history 'length' ({length}) exceeds HistoryBuffer size ({history_length})."
            )
        if sample_rate <= 0:
            raise ValueError(f"obs_history 'sample_rate' must be > 0, got {sample_rate}.")

        sliced = history_tensor[:, :length:sample_rate, :]  # [B, n_steps, D]
        n_steps = sliced.shape[1]
        expected_feature_dim = obs_info.feature_dim if obs_info.feature_dim is not None else feature_dim
        flat = sliced[:, :, :expected_feature_dim].reshape(num_envs, n_steps * expected_feature_dim)

        if flat.shape[1] != obs_info.dim:
            raise ValueError(
                f"obs_history produced {flat.shape[1]} values "
                f"(n_steps={n_steps} × feature_dim={expected_feature_dim}), "
                f"but dim={obs_info.dim} in YAML. "
                f"Ensure dim == (ceil(length / sample_rate)) * feature_dim."
            )
        return flat

    # ------------------------------------------------------------------ #
    #  Observation spaces
    # ------------------------------------------------------------------ #
    def get_policy_observation_space(self) -> gym.spaces.Box:
        return self._make_space(self.policy_obs_dim, self._policy_min, self._policy_max)

    def get_critic_observation_space(self) -> gym.spaces.Box:
        return self._make_space(self.critic_obs_dim, self._critic_min, self._critic_max)

    def get_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            "policy": self.get_policy_observation_space(),
            "critic": self.get_critic_observation_space(),
        })

    def _make_space(self, dim: int, mins: list[float], maxs: list[float]) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf if self.normalize else np.array(mins, dtype=np.float32),
            high=np.inf if self.normalize else np.array(maxs, dtype=np.float32),
            shape=(dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Normalization helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(obs: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
        denom = max_t - min_t
        return (obs - min_t) / torch.where(denom == 0, torch.ones_like(denom), denom) * 2.0 - 1.0

    # ------------------------------------------------------------------ #
    #  Main entry: returns (policy_obs, critic_obs)
    # ------------------------------------------------------------------ #
    def create_obs(
        self,
        state: torch.Tensor,
        env_metrics: dict[str, torch.Tensor],
        friction_map: torch.Tensor | None = None,
        curvature_map: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_wrapped = StateWrapper(state)._asdict()
        data_source = {**state_wrapped, **env_metrics}
        dev = state.device

        # ---------------------------------------------------------------
        # Source-level noise: noise all unique transform sources at once,
        # then cache the results.  All observations sharing the same source
        # name receive the same noisy value – e.g. sin and cos of `progress`
        # will use the same noisy angle, preserving sin²+cos²=1.
        # ---------------------------------------------------------------
        _noisy_src: dict[str, torch.Tensor] = {}
        if self._source_names:
            # Migrate persistent tensors to the obs device (once, then cached)
            if self._source_bias_t.device != dev:
                self._source_bias_t = self._source_bias_t.to(dev)
                self._source_std_t = self._source_std_t.to(dev)
                self._source_bias_std_t = self._source_bias_std_t.to(dev)

            # Stack raw sources: [num_envs, num_sources]
            raw_sources = torch.stack(
                [data_source[name] for name in self._source_names], dim=-1
            )
            # Add bias and (optionally) per-step noise in one shot
            noisy = raw_sources + self._source_bias_t
            if self._has_source_step_noise:
                noisy = noisy + torch.randn_like(raw_sources) * self._source_std_t

            # Slice into per-source [num_envs, 1] views (no copy)
            for name, idx in self._source_idx.items():
                _noisy_src[name] = noisy[:, idx : idx + 1]

        # ---------------------------------------------------------------
        # Build policy and critic component lists
        # ---------------------------------------------------------------
        policy_components: list[torch.Tensor] = []
        critic_components: list[torch.Tensor] = []

        for obs_info in self.observation_list:

            # ---- Transform-derived obs (sin/cos of a source signal) ------
            if obs_info.transform is not None:
                val = _TRANSFORMS[obs_info.transform](_noisy_src[obs_info.name])
                critic_components.append(val)
                if not obs_info.privileged:
                    policy_components.append(val)
                continue

            # ---- History obs --------------------------------------------
            if obs_info.name == "obs_history":
                if history is None:
                    raise ValueError(
                        "Observation 'obs_history' requested in YAML, but no history "
                        "was provided. Ensure exclude_history=False in the simulator."
                    )
                val = self._create_history_obs(history, obs_info)

            # ---- Friction / curvature maps (iterable) -------------------
            elif obs_info.name in ("track_friction", "friction"):
                if friction_map is None:
                    raise ValueError(
                        f"Observation '{obs_info.name}' requested in YAML, but no "
                        "friction_map was provided. Ensure exclude_friction=False."
                    )
                val = self._create_iterable_obs(friction_map, obs_info)

            elif obs_info.name == "track_curvature":
                if curvature_map is None:
                    raise ValueError(
                        "Observation 'track_curvature' requested in YAML, but no "
                        "curvature_map was provided."
                    )
                val = self._create_iterable_obs(curvature_map, obs_info)

            # ---- Regular scalar / vector obs ----------------------------
            else:
                if obs_info.name not in data_source:
                    raise KeyError(
                        f"Observation '{obs_info.name}' not found in state or metrics."
                    )
                val = data_source[obs_info.name]
                if val.dim() == 1:
                    val = val.unsqueeze(-1)

            critic_components.append(val)
            if not obs_info.privileged:
                policy_components.append(val)

        # ---------------------------------------------------------------
        # Concatenate
        # ---------------------------------------------------------------
        policy_obs = torch.cat(policy_components, dim=-1)
        critic_obs = torch.cat(critic_components, dim=-1)

        # Apply column-level noise (non-transform obs) in one vectorised pass
        if self.obs_randomizer is not None:
            policy_obs = self.obs_randomizer.apply_all(policy_obs)

        # Normalize
        if self.normalize:
            if self.policy_min_t.device != dev:
                self.policy_min_t = self.policy_min_t.to(dev)
                self.policy_max_t = self.policy_max_t.to(dev)
                self.critic_min_t = self.critic_min_t.to(dev)
                self.critic_max_t = self.critic_max_t.to(dev)

            policy_obs = self._normalize(policy_obs, self.policy_min_t, self.policy_max_t)
            critic_obs = self._normalize(critic_obs, self.critic_min_t, self.critic_max_t)

        return policy_obs, critic_obs

    def get_obs_file_path(self, name: str) -> str:
        return os.path.join(os.path.dirname(__file__), "config", f"{name}.yaml")
