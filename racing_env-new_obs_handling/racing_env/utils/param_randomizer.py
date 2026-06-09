import torch
import logging

from racing_env.utils.constants import TRACK

_FRICTION_KEYS = {"friction", "friction_step", "friction_const"}


class ParamRandomizer:
    """Randomizes per-environment model parameters stored in a ``model_params`` dict.

    The ``model_params`` dict mirrors the layout produced by
    ``LearnedModel.build_params_dict()``: keys are the module's named-buffer /
    named-parameter names, values are tensors of shape ``[num_envs]`` (or
    ``[num_envs, d]`` for array-valued buffers).

    Each key in the randomization config that is **not** a friction key must
    match a key in ``model_params``; the value is treated as a relative
    Gaussian standard deviation (sigma) and the buffer is multiplied by
    ``N(1, sigma)`` per environment.

    Friction keys (``friction``, ``friction_step``, ``friction_const``) affect
    the track friction tensor which is returned separately.
    """

    def __init__(
        self,
        rand_params: dict,
        device: torch.device,
        model_params_orig: dict[str, torch.Tensor],
        track_size: torch.Tensor,
        num_envs: int,
    ):
        self.rand_params = rand_params
        self.device = device
        self.model_params_orig = model_params_orig
        self.track_size = track_size
        self.num_envs = num_envs
        self.max_track_size: int = int(track_size.max().item())

    def randomize(
        self,
        indices: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Randomize all parameters according to the config.

        Args:
            indices: Optional 1-D integer tensor of environment indices that
                need new values.  When provided, only ``len(indices)``
                samples are generated and the returned tensors have shape
                ``[len(indices), ...]`` instead of ``[num_envs, ...]``.  The
                caller is responsible for scattering them back into the
                full-size buffers.  When *None* (default), all ``num_envs``
                environments are randomized (original behaviour).

        Returns:
            model_params: Dict with same keys as ``model_params_orig`` but
                containing only values for the requested environments.
            friction: ``[n, max_track_size]`` friction tensor (n = len(indices)
                or num_envs).
        """
        if indices is not None:
            n = len(indices)
            track_sizes = self.track_size[indices]
        else:
            n = self.num_envs
            track_sizes = self.track_size

        model_params_orig_n = (
            {k: v[indices] for k, v in self.model_params_orig.items()}
            if indices is not None
            else self.model_params_orig
        )
        model_params = dict(model_params_orig_n)  # shallow copy
        friction = torch.ones(
            (n, self.max_track_size),
            device=self.device,
        ) * TRACK.DEFAULT_FRICTION

        for param, sigma in self.rand_params.items():
            if param in _FRICTION_KEYS:
                if param == "friction":
                    friction = self._randomize_friction(friction, n, track_sizes)
                elif param == "friction_step":
                    friction = self._randomize_friction_step(friction, n, track_sizes)
                elif param == "friction_const":
                    friction = self._randomize_friction_const(friction, n)
            elif param in model_params:
                model_params[param] = self._randomize_scalar(
                    model_params[param], model_params_orig_n[param],
                    sigma, is_tau="tau" in param,
                )
            else:
                logging.error(
                    f"Randomization parameter '{param}' not found in model_params keys: "
                    f"{list(model_params.keys())}"
                )

        return model_params, friction

    # ------------------------------------------------------------------

    def _randomize_scalar(
        self,
        buf: torch.Tensor,
        buf_orig: torch.Tensor,
        sigma: float,
        is_tau: bool = False,
    ) -> torch.Tensor:
        rand = torch.normal(1.0, sigma, size=buf_orig.shape, device=self.device)
        min_r = 0.4 if is_tau else 0.1
        rand = torch.clamp(rand, min_r, 2.0)
        result = buf_orig * rand
        if torch.any(result < 0):
            logging.error(f"Randomized buffer produced negative values (sigma={sigma}).")
        return result


    def _randomize_friction(self, friction: torch.Tensor, num_envs: int, track_sizes: torch.Tensor) -> torch.Tensor:
        sigma = self.rand_params["friction"]
        device = self.device
        track_sizes = track_sizes.long()
        max_size = self.max_track_size

        # --- Gaussian kernel ---------------------------------------------------
        gauss_sigma = 5
        kernel_radius = 3 * gauss_sigma
        kernel_x = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel_x / gauss_sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)  # [1, 1, K]

        # --- Control points (fixed n_ctrl=6, matching the original max) --------
        # ext_x layout: [x0=0, interior...(4 pts), x_last=size-1] → n_ctrl=6 nodes, 5 segments
        n_interior = 4
        n_ctrl = n_interior + 2  # 6 total nodes

        # Interior x positions: sorted uniform samples in [0, size-1] per env
        ctrl_x_raw = torch.rand(num_envs, n_interior, device=device)
        ctrl_x_raw, _ = torch.sort(ctrl_x_raw, dim=1)
        ctrl_x = ctrl_x_raw * (track_sizes.float().unsqueeze(1) - 1)  # [num_envs, n_interior]

        # Full knot x-coords: [num_envs, n_ctrl]
        ext_x = torch.cat([
            torch.zeros(num_envs, 1, device=device),
            ctrl_x,
            (track_sizes.float() - 1).unsqueeze(1),
        ], dim=1)  # [num_envs, n_ctrl]

        # y-values: one per segment (n_ctrl-1=5), then wrap first to close the loop
        ctrl_y = torch.clamp(
            torch.normal(TRACK.DEFAULT_FRICTION, sigma, (num_envs, n_ctrl - 1), device=device),
            TRACK.MIN_FRICTION, TRACK.MAX_FRICTION,
        )
        ext_y = torch.cat([ctrl_y, ctrl_y[:, :1]], dim=1)  # [num_envs, n_ctrl]

        # --- Batched piecewise-linear interpolation ----------------------------
        # x_q: [num_envs, max_size]
        x_q = torch.arange(max_size, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)

        # searchsorted over the n_ctrl knots for every query point: [num_envs, max_size]
        idx = torch.searchsorted(ext_x.contiguous(), x_q.contiguous(), right=True)
        idx = idx.clamp(1, n_ctrl - 1)

        x0 = ext_x.gather(1, idx - 1)   # [num_envs, max_size]
        x1 = ext_x.gather(1, idx)
        y0 = ext_y.gather(1, idx - 1)
        y1 = ext_y.gather(1, idx)
        t = (x_q - x0) / (x1 - x0).clamp(min=1e-6)
        interp = y0 + t * (y1 - y0)     # [num_envs, max_size]

        # Fill positions beyond each env's track_size with the last valid value
        # so that circular-padding in the convolution sees a smooth boundary.
        pos_mask = torch.arange(max_size, device=device).unsqueeze(0) >= track_sizes.unsqueeze(1)
        last_vals = interp.gather(1, (track_sizes - 1).clamp(min=0).unsqueeze(1))  # [num_envs, 1]
        interp = torch.where(pos_mask, last_vals.expand_as(interp), interp)

        # --- Batched circular Gaussian smoothing -------------------------------
        # F.pad with mode='circular' wraps around the full max_size axis; for envs
        # whose track_size == max_size this is exact; for smaller tracks the
        # out-of-range region was filled with the boundary value above, so the
        # wrap-around artefacts at the true track boundary are minimal.
        padded = torch.nn.functional.pad(
            interp.unsqueeze(1), (kernel_radius, kernel_radius), mode="circular"
        )  # [num_envs, 1, max_size + 2*kernel_radius]
        smoothed = torch.nn.functional.conv1d(padded, kernel).squeeze(1)  # [num_envs, max_size]

        # Write back only valid positions
        valid_mask = ~pos_mask
        friction[:, :max_size] = torch.where(valid_mask, smoothed, friction[:, :max_size])

        return friction

    def _randomize_friction_step(self, friction: torch.Tensor, num_envs: int, track_sizes: torch.Tensor) -> torch.Tensor:
        sigma = self.rand_params["friction_step"]
        max_size = self.max_track_size
        max_patches = 7  # randint(0, 8) → 0..7
        device = self.device

        # [num_envs] number of active patches per env (0 = keep default friction)
        num_patches = torch.randint(0, 8, (num_envs,), device=device)

        # Per-env base friction, per-patch friction values [num_envs, max_patches]
        default_f = torch.rand(num_envs, device=device) * 0.2 + 0.7
        patch_f = torch.clamp(
            default_f.unsqueeze(1) + torch.normal(0.0, sigma, (num_envs, max_patches), device=device),
            TRACK.MIN_FRICTION, TRACK.MAX_FRICTION,
        )

        # Zero out weights for patches beyond num_patches, then normalise
        patch_mask = torch.arange(max_patches, device=device).unsqueeze(0) < num_patches.unsqueeze(1)
        weights = torch.rand(num_envs, max_patches, device=device) * patch_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Integer lengths; correct the last active patch so they sum exactly to track_size
        lengths = (weights * track_sizes.float().unsqueeze(1)).round().long()
        last_idx = (num_patches - 1).clamp(min=0).unsqueeze(1)         # [num_envs, 1]
        correction = (track_sizes - lengths.sum(dim=1)).unsqueeze(1)   # [num_envs, 1]
        lengths.scatter_add_(1, last_idx, correction)
        lengths = lengths.clamp(min=0)

        # Cumulative boundaries [num_envs, max_patches-1] for searchsorted
        boundaries = torch.cumsum(lengths[:, :-1], dim=1).float()

        # Map every track position to its patch index  [num_envs, max_size]
        x_q = torch.arange(max_size, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
        patch_idx = torch.searchsorted(boundaries.contiguous(), x_q.contiguous(), right=True)
        patch_idx = patch_idx.clamp(0, max_patches - 1)

        # Gather per-position friction and write back only valid, active positions
        new_f = patch_f.gather(1, patch_idx)
        valid = (x_q < track_sizes.float().unsqueeze(1)) & (num_patches > 0).unsqueeze(1)
        friction = torch.where(valid, new_f, friction)
        return friction

    def _randomize_friction_const(self, friction: torch.Tensor, num_envs: int) -> torch.Tensor:
        sigma = self.rand_params["friction_const"]
        new_f = torch.clamp(
            torch.normal(TRACK.DEFAULT_FRICTION, sigma, (num_envs,), device=self.device),
            TRACK.MIN_FRICTION, TRACK.MAX_FRICTION,
        )
        friction[:] = new_f.unsqueeze(1)
        return friction
