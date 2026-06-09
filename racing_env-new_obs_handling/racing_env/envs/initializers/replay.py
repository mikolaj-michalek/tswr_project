import torch

from racing_env.envs.initializers.zero import ZeroInitializer
from racing_env.utils.constants import STATE


class ReplayInitializer(ZeroInitializer):
    """
    Mixed initializer that splits each reset batch between two strategies:

    * ``zero_fraction`` of the environments to reset are handled by the
      standard :class:`ZeroInitializer` (random track position, minimum
      velocity, zero controls).
    * The remaining ``(1 - zero_fraction)`` environments are restored to a
      state that was previously visited during training, sampled uniformly
      from a fixed-capacity circular replay buffer.

    If the replay buffer is empty (e.g. at the very beginning of training),
    the initializer silently falls back to :class:`ZeroInitializer` for all
    environments so that training can start without any special handling.

    Usage
    -----
    The buffer must be populated during training by calling
    :meth:`record_states` after every step (or at episode boundaries)::

        initializer.record_states(
            sim.state,          # [num_envs, STATE.SIZE]
            sim.closest_idx,    # [num_envs]
            sim.last_s,         # [num_envs]
        )

    Args:
        env: The simulator / environment instance (passed to the parent).
        zero_fraction (float): Fraction in ``[0, 1]`` of reset environments
            that will be initialised with the :class:`ZeroInitializer`.
            Default ``0.5``.
        buffer_capacity (int): Maximum number of states stored in the replay
            buffer.  Older entries are overwritten in a circular fashion once
            the buffer is full.  Default ``100 000``.
    """

    def __init__(
        self,
        env,
        zero_fraction: float = 0.5,
        buffer_capacity: int = 100_000,
    ) -> None:
        super().__init__(env)

        if not 0.0 <= zero_fraction <= 1.0:
            raise ValueError(
                f"zero_fraction must be in [0, 1], got {zero_fraction}"
            )

        self.zero_fraction = zero_fraction
        self.buffer_capacity = buffer_capacity

        dev = self.env.device
        state_dim = STATE.SIZE

        # Circular replay buffer tensors (pre-allocated on the target device)
        self._state_buf = torch.zeros(
            (buffer_capacity, state_dim), dtype=torch.float32, device=dev
        )
        self._closest_idx_buf = torch.zeros(
            (buffer_capacity,), dtype=torch.int64, device=dev
        )
        self._last_s_buf = torch.zeros(
            (buffer_capacity,), dtype=torch.float32, device=dev
        )

        # Write pointer and current fill level
        self._buf_ptr: int = 0
        self._buf_size: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_states(
        self,
        states: torch.Tensor,
        closest_idx: torch.Tensor,
        last_s: torch.Tensor,
    ) -> None:
        """Add a batch of observed states to the circular replay buffer.

        Can be called every environment step or only at episode boundaries –
        both work correctly.

        Args:
            states (torch.Tensor): Full state tensor, shape ``[n, STATE.SIZE]``.
            closest_idx (torch.Tensor): Closest track-point index per env,
                shape ``[n]``.
            last_s (torch.Tensor): Arc-length progress ``s`` per env,
                shape ``[n]``.
        """
        n = states.shape[0]
        if n == 0:
            return

        cap = self.buffer_capacity
        end = self._buf_ptr + n

        if end <= cap:
            # Fits without wrapping
            self._state_buf[self._buf_ptr : end] = states.detach()
            self._closest_idx_buf[self._buf_ptr : end] = closest_idx.detach()
            self._last_s_buf[self._buf_ptr : end] = last_s.detach()
        else:
            # Split across the circular boundary
            first = cap - self._buf_ptr
            self._state_buf[self._buf_ptr :] = states[:first].detach()
            self._closest_idx_buf[self._buf_ptr :] = closest_idx[:first].detach()
            self._last_s_buf[self._buf_ptr :] = last_s[:first].detach()

            second = n - first
            self._state_buf[:second] = states[first:].detach()
            self._closest_idx_buf[:second] = closest_idx[first:].detach()
            self._last_s_buf[:second] = last_s[first:].detach()

        self._buf_ptr = end % cap
        self._buf_size = min(self._buf_size + n, cap)

    @property
    def is_buffer_empty(self) -> bool:
        """``True`` when no states have been recorded yet."""
        return self._buf_size == 0

    # ------------------------------------------------------------------
    # Core initializer logic
    # ------------------------------------------------------------------

    def initialize(self, mask: torch.Tensor | None = None, start_at_zero: bool = False) -> None:
        """Initialize environments selected by ``mask``.

        Among the environments that need to be reset (``mask == True``),
        ``zero_fraction`` are handled by :class:`ZeroInitializer` and the
        rest are restored from the replay buffer.  If the buffer is empty
        the method falls back to :class:`ZeroInitializer` for all.

        Args:
            mask (torch.Tensor | None): Boolean mask of shape ``[num_envs]``.
                ``True`` entries are reset.  ``None`` resets all environments.
            start_at_zero (bool): Forwarded to :class:`ZeroInitializer` for
                the zero-initialised subset.
        """
        dev = self.env.device
        num_envs = self.env.num_envs

        if mask is None:
            mask = torch.ones(num_envs, dtype=torch.bool, device=dev)

        # Fall back to ZeroInitializer when the buffer has not been seeded yet
        if self.is_buffer_empty:
            super().initialize(mask=mask, start_at_zero=start_at_zero)
            return

        reset_indices = mask.nonzero(as_tuple=False).squeeze(1)  # [n_reset]
        n_reset = reset_indices.shape[0]
        if n_reset == 0:
            return

        # Determine how many environments go to each strategy
        n_zero = round(self.zero_fraction * n_reset)
        n_zero = max(0, min(n_zero, n_reset))   # clamp to [0, n_reset]
        n_replay = n_reset - n_zero

        # Random assignment: shuffle which envs get which strategy
        perm = torch.randperm(n_reset, device=dev)
        zero_env_indices = reset_indices[perm[:n_zero]]
        replay_env_indices = reset_indices[perm[n_zero:]]

        # ---- Zero-initialised subset --------------------------------
        if n_zero > 0:
            zero_mask = torch.zeros(num_envs, dtype=torch.bool, device=dev)
            zero_mask[zero_env_indices] = True
            super().initialize(mask=zero_mask, start_at_zero=start_at_zero)

        # ---- Replay-initialised subset ------------------------------
        if n_replay > 0:
            sample_idx = torch.randint(
                0, self._buf_size, (n_replay,), device=dev
            )
            sampled_states = self._state_buf[sample_idx]        # [n_replay, STATE.SIZE]
            sampled_closest = self._closest_idx_buf[sample_idx]  # [n_replay]
            sampled_last_s = self._last_s_buf[sample_idx]        # [n_replay]

            self.env.state[replay_env_indices] = sampled_states
            self.env.last_s[replay_env_indices] = sampled_last_s
            self.env.closest_idx[replay_env_indices] = sampled_closest
