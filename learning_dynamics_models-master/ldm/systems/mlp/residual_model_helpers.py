"""
Model helpers for the latent-conditioned residual dynamics model.

Exports
-------
STATE_DIM, CONTROL_DIM, OBS_DIM
    Dimension constants re-exported from
    ``ldm.systems.f1tenth.utils.f1tenth_observation_preprocesor``
    (single source of truth).

HistoryPreprocessor
    Applies CarObservationPreprocessor to a (B, T, obs_dim) history tensor.

build_models(cfg) -> (CnnVAEHistoryEncoder, ResidualMlp)
    Constructs encoder and residual MLP from a config dict.

ResidualRollout
    Chunked, compilable euler/rk4 rollout for the combined base+residual model.
    Mirrors the structure of ``RolloutModel``.

rollout_free_running_loss(...) -> (total, recon, kl)
    Computes the combined reconstruction + KL loss using a free-running
    (autoregressive) rollout via ``ResidualRollout``.

estimate_lipschitz_wrt_z(residual_mlp, t, x, u, z) -> frob_sq
    Estimates the Lipschitz constant of the residual MLP w.r.t. the latent
    code z via the squared Frobenius norm of the Jacobian ∂dR/∂z, averaged
    over the batch.  The square root is a valid Lipschitz upper bound.
    Supports ``create_graph=True`` for use as a differentiable regulariser.
"""

from typing import Union

import torch
import torch.nn as nn

from ldm.systems.mlp.cnn_vae_history_encoder import (
    CnnBoundedHistoryEncoder,
    CnnVAEHistoryEncoder,
)
from ldm.systems.mlp.cnn_vae_future_encoder import (
    CnnBoundedFutureEncoder,
    CnnVAEFutureEncoder,
)
from ldm.systems.mlp.residual_mlp import ResidualMlp
from ldm.systems.f1tenth.utils.f1tenth_observation_preprocesor import (
    CarObservationPreprocessor,
    STATE_DIM,
    CONTROL_DIM,
    OBS_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────
# History preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class HistoryPreprocessor(nn.Module):
    """
    Applies CarObservationPreprocessor frame-by-frame to a history tensor.

    Args:
        M : (B, T, obs_dim)
    Returns:
        (B, T, obs_dim)  — normalised
    """

    def __init__(self):
        super().__init__()
        self.obs_prep = CarObservationPreprocessor()

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        B, T, D = M.shape
        return self.obs_prep(M.reshape(B * T, D)).reshape(B, T, D - 2) # drop rear omega wheel and friction as they are repeated in observations

# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_models(
    z_dim,
    enc_channels,
    enc_kernel,
    mlp_hidden,
    observation_window_len,
    encoder_type: str = "cvae",
):
    """
    Build encoder and residual MLP.

    z_dim                  : int   — latent code dimension
    enc_channels           : list  — CNN encoder channel sizes, e.g. [32, 64, 64]
    enc_kernel             : int   — CNN kernel size
    mlp_hidden             : list  — hidden layer widths for the residual MLP
    observation_window_len : int
    encoder_type           : str   — ``"cvae"`` or ``"bounded_ae"``

    Returns
    -------
    encoder      : CnnVAEHistoryEncoder | CnnBoundedHistoryEncoder
    residual_mlp : ResidualMlp
    """

    if encoder_type == "cvae":
        encoder = CnnVAEHistoryEncoder(
            obs_dim=OBS_DIM,
            z_dim=z_dim,
            history_len=observation_window_len,
            channels=enc_channels,
            kernel_size=enc_kernel,
        )
    elif encoder_type == "bounded_ae":
        encoder = CnnBoundedHistoryEncoder(
            obs_dim=OBS_DIM,
            z_dim=z_dim,
            history_len=observation_window_len,
            channels=enc_channels,
            kernel_size=enc_kernel,
        )
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Expected 'cvae' or 'bounded_ae'."
        )

    residual_mlp = ResidualMlp(
        state_dim    = STATE_DIM,
        control_dim  = CONTROL_DIM,
        z_dim        = z_dim,
        hidden_sizes = mlp_hidden,
        activation   = nn.ReLU(),
    )

    return encoder, residual_mlp

def build_models_from_cfg(cfg):
    """
    Build encoder and residual MLP from a config dict.

    The config dict must contain the following keys:
    - z_dim
    - enc_channels
    - enc_kernel
    - mlp_hidden
    - observation_window_len

    Returns
    -------
    encoder      : CnnVAEHistoryEncoder
    residual_mlp : ResidualMlp
    """

    return build_models(
        z_dim=cfg["z_dim"],
        enc_channels=cfg["enc_channels"],
        enc_kernel=cfg["enc_kernel"],
        mlp_hidden=cfg["mlp_hidden"],
        observation_window_len=cfg["observation_window_len"],
        encoder_type=cfg.get("history_encoder_type", "cvae"),
    )


def build_future_encoder(
    z_dim,
    enc_channels,
    enc_kernel,
    prediction_horizon,
    extra_dim: int = 0,
    encoder_type: str = "cvae",
):
    """
    Build a future-trajectory encoder.

    The encoder conditions on the ground-truth future state trajectory
    ``X_future : (B, H, state_dim)`` and produces a latent code z.  It is
    intended to be used as the posterior / recognition network during
    training. Supported modes are ``"cvae"`` and ``"bounded_ae"``.

    Parameters
    ----------
    z_dim              : int   — latent code dimension
    enc_channels       : list  — CNN channel sizes, e.g. [32, 64, 64]
    enc_kernel         : int   — CNN kernel size
    prediction_horizon : int   — number of future steps H
    extra_dim          : int   — extra per-step features appended to the state
                                 before the CNN (e.g. ctrl_dim, state_dim for
                                 nominal rollout, or their sum for both)
    encoder_type       : str   — ``"cvae"`` or ``"bounded_ae"``

    Returns
    -------
    future_encoder : CnnVAEFutureEncoder | CnnBoundedFutureEncoder
    """
    if encoder_type == "cvae":
        return CnnVAEFutureEncoder(
            state_dim=STATE_DIM,
            z_dim=z_dim,
            horizon=prediction_horizon,
            channels=enc_channels,
            kernel_size=enc_kernel,
            extra_dim=extra_dim,
        )
    if encoder_type == "bounded_ae":
        return CnnBoundedFutureEncoder(
            state_dim=STATE_DIM,
            z_dim=z_dim,
            horizon=prediction_horizon,
            channels=enc_channels,
            kernel_size=enc_kernel,
            extra_dim=extra_dim,
        )
    raise ValueError(
        f"Unknown future encoder type: {encoder_type}. Expected 'cvae' or 'bounded_ae'."
    )


def build_future_encoder_from_cfg(cfg):
    """
    Build a future-trajectory encoder from a config dict.

    The config dict must contain the following keys:
    - z_dim
    - enc_channels
    - enc_kernel
    - prediction_horizon

    Returns
    -------
    future_encoder : CnnVAEFutureEncoder | CnnBoundedFutureEncoder
    """
    return build_future_encoder(
        z_dim=cfg["z_dim"],
        enc_channels=cfg["enc_channels"],
        enc_kernel=cfg["enc_kernel"],
        prediction_horizon=cfg["prediction_horizon"],
        extra_dim=cfg.get("future_encoder_extra_dim", 0),
        encoder_type=cfg.get("future_encoder_type", "cvae"),
    )


def compute_nominal_rollout(
    base_model:         nn.Module,
    X0:                 torch.Tensor,  # (B, state_dim)
    U:                  torch.Tensor,  # (B, H, ctrl_dim)
    Tp:                 float,
    integration_method: str = "euler",
) -> torch.Tensor:
    """
    Roll out the **frozen base model** for ``H`` steps and return the predicted
    state trajectory.

    Parameters
    ----------
    base_model         : frozen nn.Module  (no gradients)
    X0                 : (B, state_dim)    initial state
    U                  : (B, H, ctrl_dim)  control sequence
    Tp                 : float             integration time-step [s]
    integration_method : ``"euler"`` or ``"rk4"``

    Returns
    -------
    X_nom : (B, H, state_dim)   nominal predicted states (no residual)
    """
    assert integration_method in ("euler", "rk4"), \
        f"Unknown integration method: {integration_method}"

    B, H, _ = U.shape
    state_dim = X0.shape[-1]
    X_nom = torch.zeros(B, H, state_dim, device=X0.device, dtype=X0.dtype)
    vx_min = torch.tensor(
        [0.5] + [float("-inf")] * (state_dim - 1), device=X0.device
    )

    x = X0
    with torch.no_grad():
        for i in range(H):
            t_now = torch.tensor([i * Tp], device=X0.device)
            u_now = U[:, i]
            if integration_method == "rk4":
                def _f(xx):
                    dx, _, _ = base_model(t_now, xx.clone(), u_now)
                    return nn.functional.pad(dx, (0, state_dim - dx.shape[-1]))
                k1 = _f(x)
                k2 = _f(x + k1 * Tp / 2)
                k3 = _f(x + k2 * Tp / 2)
                k4 = _f(x + k3 * Tp)
                x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Tp
            else:
                dx, _, _ = base_model(t_now, x.clone(), u_now)
                dx = nn.functional.pad(dx, (0, state_dim - dx.shape[-1]))
                x_next = x + dx * Tp
            x_next = torch.clamp(x_next, min=vx_min)
            X_nom[:, i] = x_next
            x = x_next

    return X_nom


# ─────────────────────────────────────────────────────────────────────────────
# Residual rollout  (mirrors RolloutModel — chunked, compilable, euler/rk4)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualRollout(nn.Module):
    """
    Free-running rollout for the combined base + residual model.

    Mirrors the structure of ``RolloutModel``:
    - ``_euler_step`` / ``_rk4_step`` — single integration steps
    - ``_forward_n_step``            — inner fixed-horizon loop (compilable)
    - ``forward``                    — chunked public entry point

    The base model is called **without** gradient tracking (its parameters
    must already have ``requires_grad=False``), so gradients flow only
    through ``residual_mlp``.

    Parameters
    ----------
    residual_mlp       : ResidualMlp
    base_model         : nn.Module   — frozen base dynamics model
    Tp                 : float       — integration time-step [s]
    integration_method : str         — ``"euler"`` or ``"rk4"``
    compile_inner      : bool        — torch.compile the inner loop
    """

    def __init__(
        self,
        residual_mlp:       ResidualMlp,
        base_model:         nn.Module,
        Tp:                 float,
        integration_method: str  = "euler",
        compile_inner:      bool = False,
    ):
        super().__init__()
        self.residual_mlp = residual_mlp
        self.base_model   = base_model
        self.Tp           = Tp

        assert integration_method in ("euler", "rk4"), \
            f"Unknown integration method: {integration_method}"
        self.integration_method = integration_method
        self._step = self._rk4_step if integration_method == "rk4" else self._euler_step

        if compile_inner:
            self._forward_n_step = torch.compile(
                self._forward_n_step,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs",
            )

    # ------------------------------------------------------------------
    # Combined derivative  dx_base (no grad) + dR (with grad)
    # Note: base_model params must have requires_grad=False so that
    #       no gradient flows through them even without no_grad().
    #       We avoid no_grad() here so the inner loop stays graph-break-free
    #       and remains compilable.
    # ------------------------------------------------------------------

    def _dx(self, t, x, u, z):
        dx_base, tf, slips = self.base_model(t, x.clone(), u)
        dx_base = nn.functional.pad(dx_base, (0, x.shape[-1] - dx_base.shape[-1]))
        dR = self.residual_mlp(t, x, u, z)
        return dx_base + dR, tf, slips

    # ------------------------------------------------------------------
    # Integration steps
    # ------------------------------------------------------------------

    def _euler_step(self, t, x, u, z):
        dx, tf, slips = self._dx(t, x, u, z)
        return x + dx * self.Tp, tf, slips

    def _rk4_step(self, t, x, u, z):
        k1, tf1, s1 = self._dx(t, x.clone(),                    u, z)
        k2, tf2, s2 = self._dx(t, x.clone() + k1 * self.Tp / 2, u, z)
        k3, tf3, s3 = self._dx(t, x.clone() + k2 * self.Tp / 2, u, z)
        k4, tf4, s4 = self._dx(t, x.clone() + k3 * self.Tp,     u, z)
        tf    = (tf1 + 2. * tf2 + 2. * tf3 + tf4) / 6.
        slips = (s1  + 2. * s2  + 2. * s3  + s4)  / 6.
        dx    = (k1  + 2. * k2  + 2. * k3  + k4)  / 6.
        return x + dx * self.Tp, tf, slips

    # ------------------------------------------------------------------
    # Inner fixed-horizon loop  (potentially compiled)
    # ------------------------------------------------------------------

    def _forward_n_step(
        self,
        z:                 torch.Tensor,  # (B, z_dim)
        X0:                torch.Tensor,  # (B, state_dim)
        U:                 torch.Tensor,  # (B, n, ctrl_dim)
        prediction_horizon: int,
    ):
        """
        Autoregressive rollout for ``prediction_horizon`` steps.

        Returns
        -------
        X_return    : (B, prediction_horizon, state_dim)
        tire_forces : accumulated scalar / tensor
        """
        X_return    = torch.zeros(
            X0.shape[0], prediction_horizon, X0.shape[-1], device=X0.device
        )
        tire_forces = 0.
        vx_min      = torch.tensor(
            [0.5] + [float("-inf")] * (X0.shape[-1] - 1), device=X0.device
        )

        for i in range(1, prediction_horizon + 1):
            t_now        = torch.tensor([i * self.Tp], device=X0.device)
            X_next, tf_, _ = self._step(t_now, X0, U[:, i - 1], z)
            X_next       = torch.clamp(X_next, min=vx_min)
            X_return[:, i - 1] = X_next
            X0          = X_next
            tire_forces = tire_forces + tf_

        return X_return, tire_forces

    # ------------------------------------------------------------------
    # Public chunked forward  (same signature logic as RolloutModel)
    # ------------------------------------------------------------------

    def forward(
        self,
        z:                 torch.Tensor,  # (B, z_dim)
        X0:                torch.Tensor,  # (B, state_dim)
        U:                 torch.Tensor,  # (B, H, ctrl_dim)
        prediction_horizon: int,
        chunk_size:         int = 1,
    ):
        """
        Chunked free-running rollout.

        Returns
        -------
        X_pred      : (B, prediction_horizon, state_dim)
        tire_forces : list of per-chunk tire-force tensors
        """
        assert prediction_horizon % chunk_size == 0, (
            f"prediction_horizon {prediction_horizon} must be divisible by chunk_size {chunk_size}"
        )
        num_chunks = prediction_horizon // chunk_size

        X_sim = torch.zeros(
            X0.shape[0], prediction_horizon + 1, X0.shape[-1], device=X0.device
        )
        X_sim[:, 0] = X0
        tire_forces = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx   = min(start_idx + chunk_size, prediction_horizon)
            current_n = end_idx - start_idx

            X_chunk, tf_ = self._forward_n_step(
                z, X_sim[:, start_idx], U[:, start_idx:end_idx], current_n
            )
            tire_forces.append(tf_)
            X_sim[:, start_idx + 1:end_idx + 1] = X_chunk

        return X_sim[:, 1:], tire_forces


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def rollout_free_running_loss(
    encoder:          CnnVAEHistoryEncoder,
    residual_rollout: "ResidualRollout",
    M:         torch.Tensor,       # (B, T, obs_dim)
    U:         torch.Tensor,       # (B, H, ctrl_dim)
    X0:        torch.Tensor,       # (B, 1, state_dim)
    X:         torch.Tensor,       # (B, H, state_dim)  — true next states
    kl_weight: float,
    loss_fn:   nn.Module,
    hist_prep: "HistoryPreprocessor",
    chunk_size: int = 1,
):
    """
    Free-running (autoregressive) rollout loss.

    Uses ``ResidualRollout`` internally, which supports euler/rk4 integration
    and can be chunk-compiled — matching the structure of ``RolloutModel``.

    Steps
    -----
    1. Encode history M → (z, mu, log_var) via the CNN VAE encoder.
    2. Free-running chunked rollout via ``residual_rollout``.
    3. Reconstruction loss = loss_fn(X_pred, X_true).
    4. KL loss = KL( q(z|M) || N(0,I) ).
    5. total = recon + kl_weight * kl.

    Parameters
    ----------
    chunk_size : int
        Chunk size for the inner rollout (must divide prediction_horizon).
        Larger chunks amortise Python overhead; set >1 when ``compile_inner``
        was used in ``ResidualRollout``.

    Returns
    -------
    total, recon, kl  — scalar tensors
    """
    B, H, _ = X.shape

    # 1. Encode history
    M_prep = hist_prep(M)                 # (B, T, obs_dim) normalised
    z, mu, log_var = encoder(M_prep)      # z: (B, z_dim)

    # 2. Chunked free-running rollout
    X0_sq = X0.squeeze(1)                 # (B, state_dim)
    X_pred, _ = residual_rollout(z, X0_sq, U, prediction_horizon=H, chunk_size=chunk_size)
    # X_pred: (B, H, state_dim)

    # 3. Reconstruction loss on total predicted state
    recon = loss_fn(X_pred, X)

    # 4. Latent regularisation
    if getattr(encoder, "uses_kl_regularization", True):
        kl = CnnVAEHistoryEncoder.kl_loss(mu, log_var)
    else:
        kl = recon.new_zeros(())

    total = recon + kl_weight * kl
    return total, recon, kl


def rollout_free_running_loss_with_future_encoder(
    future_encoder:   Union[CnnVAEFutureEncoder, CnnBoundedFutureEncoder],
    residual_rollout: "ResidualRollout",
    U:         torch.Tensor,              # (B, H, ctrl_dim)
    X0:        torch.Tensor,              # (B, 1, state_dim)
    X:         torch.Tensor,              # (B, H, state_dim)  — true next states
    kl_weight: float,
    loss_fn:   nn.Module,
    chunk_size: int = 1,
    U_future:   torch.Tensor = None,      # (B, H, ctrl_dim)   — optional control conditioning
    X_nom:      torch.Tensor = None,      # (B, H, state_dim)  — optional nominal rollout conditioning
):
    """
    Free-running (autoregressive) rollout loss using a **future-trajectory
    encoder** as the posterior / recognition network.

    During training the encoder sees the ground-truth future states, producing
    a richer latent code z than the prior.  At inference time the caller can
    substitute a history encoder or set z = 0.  In CVAE mode this is the
    standard variational training objective; in bounded-AE mode the latent
    code is deterministic and bounded to ``[-1, 1]^N``.

    The encoder input is built by concatenating along the feature axis::

        enc_input = cat([X,                     # always  (B, H, state_dim)
                         U_future,              # optional (B, H, ctrl_dim)
                         X_nom,                 # optional (B, H, state_dim)
                        ], dim=-1)

    The ``future_encoder`` must have been built with a matching ``extra_dim``
    (= ``ctrl_dim + state_dim`` if both options are used).

    Steps
    -----
    1. Build encoder input by concatenating available features.
    2. Encode → ``(z, aux_1, aux_2)`` via the selected future encoder.
    3. Free-running chunked rollout via ``residual_rollout`` conditioned on z.
    4. Reconstruction loss = loss_fn(X_pred, X_true).
    5. Latent regularisation = KL for CVAE, zero for bounded AE.
    6. total = recon + kl_weight * latent_reg.

    Parameters
    ----------
    future_encoder : CnnVAEFutureEncoder | CnnBoundedFutureEncoder
        Must be built with ``extra_dim = ctrl_dim`` (controls only),
        ``extra_dim = state_dim`` (nominal only), or
        ``extra_dim = state_dim + ctrl_dim`` (both).
    residual_rollout : ResidualRollout
        Combined base + residual dynamics rollout.
    U          : (B, H, ctrl_dim)   — control sequence (used for rollout)
    X0         : (B, 1, state_dim)  — initial state
    X          : (B, H, state_dim)  — ground-truth future states
    kl_weight  : float              — KL annealing coefficient
    loss_fn    : nn.Module          — reconstruction loss (MSE or L1)
    chunk_size : int                — chunk size for inner rollout
    U_future   : (B, H, ctrl_dim) or None— future controls appended to encoder input
    X_nom      : (B, H, state_dim) or None— nominal model rollout appended to encoder input

    Returns
    -------
    total, recon, latent_reg  — scalar tensors
    """
    B, H, _ = X.shape

    # 1. Build encoder input
    enc_parts = [X]
    if U_future is not None:
        enc_parts.append(U_future)
    if X_nom is not None:
        enc_parts.append(X_nom)
    enc_input = torch.cat(enc_parts, dim=-1)   # (B, H, state_dim [+ ctrl_dim] [+ state_dim])

    # 2. Encode (posterior / recognition network)
    z, aux_1, aux_2 = future_encoder(enc_input)  # z: (B, z_dim)

    # 3. Chunked free-running rollout
    X0_sq = X0.squeeze(1)                 # (B, state_dim)
    X_pred, _ = residual_rollout(z, X0_sq, U, prediction_horizon=H, chunk_size=chunk_size)
    # X_pred: (B, H, state_dim)

    # 4. Reconstruction loss
    recon = loss_fn(X_pred, X)

    # 5. Latent regularisation
    if getattr(future_encoder, "uses_kl_regularization", True):
        latent_reg = CnnVAEFutureEncoder.kl_loss(aux_1, aux_2)
    else:
        latent_reg = recon.new_zeros(())

    total = recon + kl_weight * latent_reg
    return total, recon, latent_reg


# ─────────────────────────────────────────────────────────────────────────────
# Lipschitz monitoring / regularisation w.r.t. z
# ─────────────────────────────────────────────────────────────────────────────

def estimate_lipschitz_wrt_z(
    residual_mlp: ResidualMlp,
    t:            torch.Tensor,   # (1,)
    x:            torch.Tensor,   # (B, state_dim)
    u:            torch.Tensor,   # (B, ctrl_dim)
    z:            torch.Tensor,   # (B, z_dim)
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Estimate the Lipschitz constant of ``residual_mlp`` with respect to the
    latent code ``z`` via the **squared Frobenius norm of the Jacobian**
    ``∂dR/∂z``, averaged over the batch.

    The Frobenius norm satisfies

        ||J||_F  ≥  sigma_max(J)

    so ``sqrt(frob_sq)`` is a valid (conservative) Lipschitz upper bound.

    The computation iterates over the ``state_dim`` output dimensions, each
    requiring one backward pass through the residual MLP — cheap for the
    typical F1tenth ``state_dim = 5``.

    Parameters
    ----------
    residual_mlp : ResidualMlp
    t            : (1,)           — time (scalar)
    x            : (B, state_dim) — states
    u            : (B, ctrl_dim)  — controls
    z            : (B, z_dim)     — latent codes; detached internally so
                                    the function is safe even when called
                                    with a z that already has gradients.
    create_graph : bool
        If ``True`` the returned tensor has a computation graph suitable for
        second-order differentiation (i.e. can be used directly as a
        regularisation term and back-propped through).  Set ``False`` for
        monitoring-only calls to avoid the extra overhead.

    Returns
    -------
    frob_sq : scalar tensor
        Mean over the batch of ``||∂dR/∂z||_F^2``.
        Take ``frob_sq.sqrt()`` for the Lipschitz estimate itself.
    """
    # Detach z so we get a clean leaf for the inner gradient computation;
    # the gradient w.r.t. residual_mlp parameters flows via create_graph.
    z_leaf = z.detach().requires_grad_(True)

    dR = residual_mlp(t, x, u, z_leaf)   # (B, state_dim)
    B, S = dR.shape

    frob_sq = z_leaf.new_zeros(())        # scalar, on correct device
    for s in range(S):
        # grad of dR[:, s].sum() w.r.t. z_leaf → (B, z_dim) = ∂dR_s/∂z per sample
        (grad_s,) = torch.autograd.grad(
            dR[:, s].sum(),
            z_leaf,
            create_graph=create_graph,
            retain_graph=True,
        )
        frob_sq = frob_sq + (grad_s ** 2).sum(dim=-1).mean()

    return frob_sq
