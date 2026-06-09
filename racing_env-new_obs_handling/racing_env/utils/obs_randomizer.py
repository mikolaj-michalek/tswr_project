import torch


class ObsRandomizer:
    """Applies per-observation white noise and per-env biases to policy observations.

    Built automatically by :class:`~racing_env.envs.obs.obs_creator.ObservationCreator`
    from the observation config YAML.  Each observation entry can declare two
    optional noise fields (in the same physical units as the raw signal, i.e.
    **before** normalization):

    * ``std``       – standard deviation of zero-mean white noise added fresh
                      every step.
    * ``bias_std``  – standard deviation used to sample a per-environment
                      constant offset once per episode.  The offset is
                      resampled (for the appropriate envs) whenever
                      :meth:`reset_biases` is called.

    Only raw (non-transform-derived) policy observations are handled here.
    Observations that declare a ``transform`` are noised at the source level
    by :class:`~racing_env.envs.obs.obs_creator.ObservationCreator` before the
    transform is applied, so that correlated observations (e.g. sin/cos of the
    same underlying signal) remain physically consistent.

    All operations are fully vectorised and stay on the observation device with
    no CPU↔GPU synchronisation in the hot path.
    """

    def __init__(
        self,
        observation_list: list,  # list[ObservationInfo] – avoid circular import
        num_envs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_envs = num_envs
        self.device = device

        # Expand each obs entry to its full column width (dim columns per obs).
        # Skip privileged obs and transform-derived obs – both are excluded from
        # the policy column-noise pass (transform obs are noised at source level).
        additive_cols: list[float] = []
        bias_cols: list[float] = []
        for obs in observation_list:
            if obs.privileged or getattr(obs, "transform", None) is not None:
                continue
            additive_cols.extend([getattr(obs, "std", 0.0) or 0.0] * obs.dim)
            bias_cols.extend([getattr(obs, "bias_std", 0.0) or 0.0] * obs.dim)

        D = len(additive_cols)

        additive_t = torch.tensor(additive_cols, dtype=torch.float32, device=device)
        self._additive_std_t = additive_t.unsqueeze(0)  # [1, D]
        self._has_additive = bool(additive_t.any())

        bias_std_t = torch.tensor(bias_cols, dtype=torch.float32, device=device)
        self._bias_std_t = bias_std_t.unsqueeze(0)  # [1, D] – for broadcasting in reset
        self._has_bias = bool(bias_std_t.any())

        # [num_envs, D] – sampled once per episode, updated via reset_biases
        if self._has_bias:
            self._bias_t = torch.randn(num_envs, D, device=device) * bias_std_t
        else:
            self._bias_t = torch.zeros(num_envs, D, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_all(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Apply column-level noise to the full concatenated policy observation tensor.

        Args:
            policy_obs: ``[num_envs, policy_dim]`` tensor (pre-normalization).

        Returns:
            Noisy tensor of the same shape (never modifies the input in-place).
        """
        dev = policy_obs.device
        if self._bias_t.device != dev:
            self._migrate(dev)

        result = policy_obs
        if self._has_bias:
            result = result + self._bias_t
        if self._has_additive:
            result = result + torch.randn_like(result) * self._additive_std_t
        return result

    def reset_biases(self, mask: torch.Tensor | None = None) -> None:
        """Resample per-env biases for the selected environments.

        Uses ``torch.where`` to avoid any GPU→CPU synchronisation.

        Args:
            mask: Boolean ``[num_envs]`` tensor.  ``None`` resets all envs.
        """
        if not self._has_bias:
            return
        dev = self._bias_t.device
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=dev)
        elif mask.device != dev:
            mask = mask.to(dev)

        # Generate candidates for all envs; select with mask – no GPU sync needed.
        new_biases = torch.randn(self.num_envs, self._bias_t.shape[1], device=dev) * self._bias_std_t
        self._bias_t = torch.where(mask.unsqueeze(-1), new_biases, self._bias_t)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _migrate(self, dev: torch.device) -> None:
        self._bias_t = self._bias_t.to(dev)
        self._bias_std_t = self._bias_std_t.to(dev)
        self._additive_std_t = self._additive_std_t.to(dev)
        self.device = dev
