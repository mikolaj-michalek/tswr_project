import torch
import torch.func
from racing_env.envs.simulators.robot_models.actuators_params import ActuatorsParameters
from racing_env.utils.state_wrapper import StateWrapper

from ldm.utils.loading import load_learned_model


class LearnedModel(torch.nn.Module):
    """Wrapper around an LDM-loaded dynamics model.

    Wrapper that makes every parameter/buffer — both the scalar physical
    constants registered on this module *and* any ``nn.Parameter`` / buffer on
    the inner ``self.learned_model`` submodule — available for per-environment
    randomisation via ``torch.func.functional_call``.

    ``build_params_dict`` iterates ``named_buffers()`` and ``named_parameters()``
    which are **recursive by default**, so inner-model entries appear with
    dotted names such as ``"learned_model.By"``, ``"learned_model.fc1.weight"``
    etc.

    * **Scalar parameters** (shape ``[1]`` or ``[]`` — e.g. Pacejka tire
      coefficients, physical constants): flattened and expanded to
      ``[num_envs]`` so they broadcast naturally against per-env state slices
      without an extra trailing dimension.
    * **Multi-dimensional NN weights** (e.g. ``nn.Linear`` weight ``[out, in]``):
      included in the dict but injecting a batched ``[num_envs, out, in]``
      weight breaks ``F.linear``.  Per-environment NN-weight randomisation
      requires wrapping the inner-model call with ``vmap``; for now such params
      are shared across environments in practice (randomised noise is applied
      identically to every env's copy).

    ``forward(t, x)`` reads parameters directly from ``self.*`` — no positional
    parameter tensors are passed.
    """

    def __init__(self, config_path: str) -> None:
        super().__init__()

        self.learned_model, model_cfg = load_learned_model(config_path)

        self.actuators_params = ActuatorsParameters()
        # Register scalar physical constants (lr, L, tau_omega, …) exposed by
        # the loaded model as named buffers on *this* module so they appear
        # with plain names (e.g. "tau_omega") in build_params_dict rather than
        # with the "learned_model." prefix.  Any nn.Parameter already living
        # on self.learned_model (e.g. Pacejka coefficients stored as Parameter)
        # is captured automatically by the recursive named_parameters() call in
        # build_params_dict — no extra registration needed here for those.
        physical_params: dict = {}
        if hasattr(self.learned_model, "get_physical_params"):
            physical_params = self.learned_model.get_physical_params()
        elif hasattr(model_cfg, "physical_params"):
            physical_params = dict(model_cfg.physical_params)

        for name, value in physical_params.items():
            self.register_buffer(name, torch.tensor([float(value)]))

    # ------------------------------------------------------------------

    def build_params_dict(self, num_envs: int) -> dict[str, torch.Tensor]:
        """Return all named buffers/parameters expanded to ``[num_envs]``.

        ``named_buffers()`` and ``named_parameters()`` are recursive, so this
        includes both the scalar buffers registered on this wrapper (plain
        names like ``"tau_omega"``) **and** every buffer/parameter belonging to
        ``self.learned_model`` (dotted names like ``"learned_model.By"``).

        Scalar tensors (``ndim <= 1``, e.g. shape ``[1]`` or ``[]``) are
        expanded to ``[num_envs]``.  Multi-dimensional tensors (NN weight
        matrices etc.) are expanded to ``[num_envs, *original_shape]`` —
        per-env behaviour for those requires ``vmap`` (see class docstring).

        Used to initialise ``Simulator.model_params`` and as the base for
        ``ParamRandomizer``.
        """
        d: dict[str, torch.Tensor] = {}
        for name, t in (*self.named_buffers(), *self.named_parameters()):
            flat = t.detach().reshape(-1)  # always at least 1-D
            if flat.numel() == 1:
                # Scalar: expand to [num_envs]
                d[name] = flat.expand(num_envs).clone()
            else:
                # Multi-dim: expand to [num_envs, *original_shape]
                d[name] = t.detach().expand(num_envs, *t.shape).clone()
        return d

    # ------------------------------------------------------------------

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """Compute state derivatives.

        When called via ``torch.func.functional_call`` with a per-env
        ``model_params`` dict, all scalar buffers (``tau_omega``, ``tau_delta``,
        ``lr``, ``L``, …) are already injected as batched ``[num_envs]`` tensors.

        Args:
            t: Current time (passed through to the inner model).
            x: State tensor ``[batch, state_dim]``.

        Returns:
            State-derivative tensor with the same shape as ``x``.
        """
        wx = StateWrapper(x)

        # Build control vector for the inner model
        u = torch.stack([wx.omega_wheels, wx.omega_wheels, wx.delta], dim=-1)
        x_model = torch.stack([wx.v_x, wx.v_y, wx.r, wx.front_friction, wx.rear_friction], dim=-1)

        dx_model, tire_forces, slips = self.learned_model(t, x_model, u)
        v_x_dot = dx_model[..., 0]
        v_y_dot = dx_model[..., 1]
        r_dot   = dx_model[..., 2]

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)

        self.last_tire_dynamics = {
            "Fx_f": Fx_f, "Fx_r": Fx_r, "Fy_f": Fy_f, "Fy_r": Fy_r,
            "alpha_f": slips[..., 0], "slip_ratio_f": slips[..., 1],
            "alpha_r": slips[..., 2], "slip_ratio_r": slips[..., 3],
        }

        # Actuator dynamics — tau buffers are [num_envs] after functional_call
        omega_wheels_dot = (wx.omega_wheels_ref - wx.omega_wheels) / self.actuators_params.tau_omega
        delta_dot        = (wx.delta_ref - wx.delta) / self.actuators_params.tau_delta

        x_dot   = wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw)
        y_dot   = wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw)
        yaw_dot = wx.r

        zeros = torch.zeros_like(wx.front_friction)
        return torch.stack([
            x_dot, y_dot, yaw_dot,
            v_x_dot, v_y_dot, r_dot,
            omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot,
            zeros, zeros, zeros, zeros,
        ], dim=1)
