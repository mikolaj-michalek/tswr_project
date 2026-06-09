"""
Training script for the latent-conditioned residual dynamics model.

Architecture overview
---------------------

  History M (B, T, obs_dim)
      |
  CnnVAEHistoryEncoder
       |
  z (B, z_dim)  +  mu / log_var   --> KL loss
       |
  +----+----------------------------------------------------------------+
  |  Free-running rollout for i in [0, H):                              |
  |    x_0  = X0                                                        |
  |    dx_base_i = base_model(x_i, u_i)          [frozen]              |
  |    dR_i      = ResidualMlp([x_i; u_i; z])                          |
  |    x_{i+1}   = x_i + (dx_base_i + dR_i) * Tp                      |
  +--------------------------------------------------------------------+
       |
  Reconstruction loss:  MSE/L1( x_pred_1..H, X_true_1..H )
       |
  Total loss = reconstruction_loss + beta * kl_loss

The free-running rollout matches test-time behaviour: the residual model
always sees its own previous predictions, not ground-truth states.

Dataset
-------
Uses the standard F1tenth CSV dataset (no preprocessing required).
Set ``base_model_path`` to point to a trained base model .pt file.

When ``use_future_encoder=True``, the future encoder can be either:
- ``cvae``: stochastic posterior with KL regularisation
- ``bounded_ae``: deterministic conditioned autoencoder with ``z ∈ [-1, 1]^N``

Usage
-----
    python experiments/train_residual.py
    python experiments/train_residual.py --z_dim 8 --lr 1e-3 --n_epochs 500
"""

import os
import sys
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_launcher import single_experiment, run_experiment

from ldm.systems.mlp.residual_model_helpers import (
    HistoryPreprocessor,
    build_models,
    build_future_encoder,
    compute_nominal_rollout,
    ResidualRollout,
    rollout_free_running_loss,
    rollout_free_running_loss_with_future_encoder,
    estimate_lipschitz_wrt_z,
)
from ldm.utils.loading import load_dynamics_model
from ldm.utils.read_datasets import read_datasets
from ldm.systems.mlp.residual_model_helpers import STATE_DIM, CONTROL_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_BASE_MODEL_PATH = str(
    Path(__file__).parent / "paper" / "f1tenth_newer_refactor" / "prediction_models" /
    "pacejka_single_track_None_gc0.01_lr0.0005_hf100_100_hb100_str2_euler_nn32_seed0.pt"
)
# ---------------------------------------------------------------------------
# Experiment  (all knobs are keyword arguments -> settable from the console)
# ---------------------------------------------------------------------------

@single_experiment
def experiment(
    # --- data ---
    dataset:                str   = "new",
    observation_window_len: int   = 100,
    prediction_horizon:     int   = 100,
    stride:                 int   = 3,
    shift_u1:               int   = 0,
    shift_u2:               int   = 0,

    # --- base model (frozen) ---
    base_model_path: str = _DEFAULT_BASE_MODEL_PATH,

    # --- residual model ---
    z_dim:        int  = 2,
    enc_channels: list = None,   # default: [32, 64, 64]
    enc_kernel:   int  = 5,
    mlp_hidden:   list = None,   # default: [64, 64]
    use_future_encoder: bool = True,  # use future trajectory encoder instead of history encoder
    #future_encoder_type: str = "cvae",  # "cvae" or "bounded_ae"
    future_encoder_type: str = "bounded_ae",  # "cvae" or "bounded_ae"
    history_encoder_type: str = "cvae",  # "cvae" or "bounded_ae" (only when use_future_encoder=False)
    condition_on_controls:        bool = False,  # also condition future encoder on U
    condition_on_nominal_rollout: bool = False,  # also condition future encoder on base-model rollout

    # --- training ---
    Tp:                 float = 0.01,
    integration_method: str   = "euler",
    chunk_size:         int   = 20,
    compile_rollout:    bool  = True,
    n_epochs:           int   = 1000,
    batch_size:         int   = 512,
    lr:                 float = 3e-4,
    grad_clip:          float = 1.0,
    kl_beta:            float = 3e-3,
    kl_warmup_epochs:   int   = 20,
    loss_fn:            str   = "mse",
    lipschitz_lambda:   float = 0.0,   # weight for ||∂dR/∂z||_F^2 regularisation (0 = disabled)
    lipschitz_n_samples: int  = 64,    # batch sub-sample size for Lipschitz estimate
    lipschitz_log_freq:  int  = 10,    # log Lipschitz estimate every N epochs

    # --- misc ---
    device:         str  = "cpu",
    seed:           int  = 0,
    wandb_project:  str  = "ldm_f1tenth_residual",
    wandb_disabled: bool = False,

    results_dir: str = "./results",
):
    # mutable list defaults
    if enc_channels is None:
        enc_channels = [32, 64, 64]
    if mlp_hidden is None:
        mlp_hidden = [64, 64]

    cfg = {**locals()}   # plain dict forwarded to build_models / wandb / checkpoints

    torch.manual_seed(seed)

    future_mode_tag = (
        f"future{future_encoder_type}" if use_future_encoder else "history"
    )
    group_name = (
        f"residual_pacejka_CAE_bs{batch_size}_lr{lr}_z{z_dim}_kl{kl_beta}_{future_mode_tag}"
        #f"residual_pacejka_bs{batch_size}_lr{lr}_z{z_dim}_kl{kl_beta}_futureU"
        #f"residual_pacejka_bs{batch_size}_lr{lr}_z{z_dim}_kl{kl_beta}_futureUXnom"
    )
    save_dir = str(Path(__file__).parent / "results" / group_name)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project = wandb_project,
        group   = group_name,
        name    = f"{group_name}_seed{seed}",
        config  = cfg,
        mode    = "disabled" if wandb_disabled else "online",
    )

    # ------------------------------------------------------------------
    # Base model (frozen)
    # ------------------------------------------------------------------
    log.info(f"Loading base model from {base_model_path} ...")
    base_model, _ = load_dynamics_model(base_model_path)
    base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    log.info("  Base model frozen.")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    log.info("Loading datasets ...")
    train_loader, val_loader, _ = read_datasets(
        "f1tenth", batch_size, observation_window_len, prediction_horizon,
        stride=stride, device=device, shift_u1=shift_u1, shift_u2=shift_u2,
        dataset=dataset,
    )
    log.info(f"  train batches: {len(train_loader)}  |  val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Residual models
    # ------------------------------------------------------------------
    # When use_future_encoder=True the history encoder is not used: z is
    # provided manually at test time via CombinedResidualDynamicsModel.set_z().
    if use_future_encoder:
        encoder = None
        _, residual_mlp = build_models(
            z_dim, enc_channels, enc_kernel, mlp_hidden, observation_window_len
        )
        cfg["history_encoder_type"] = "none"
        residual_mlp = residual_mlp.to(device)
        extra_dim = (
            (CONTROL_DIM if condition_on_controls else 0) +
            (STATE_DIM   if condition_on_nominal_rollout else 0)
        )
        future_encoder = build_future_encoder(
            z_dim, enc_channels, enc_kernel, prediction_horizon,
            extra_dim=extra_dim,
            encoder_type=future_encoder_type,
        ).to(device)
        cfg["future_encoder_extra_dim"] = extra_dim
        log.info(
            f"Using FUTURE trajectory encoder type={future_encoder_type}. History encoder disabled. "
            f"Extra conditioning: controls={condition_on_controls}, "
            f"nominal_rollout={condition_on_nominal_rollout}, extra_dim={extra_dim}."
        )
        log.info(future_encoder)
    else:
        encoder, residual_mlp = build_models(
            z_dim, enc_channels, enc_kernel, mlp_hidden, observation_window_len,
            encoder_type=history_encoder_type,
        )
        encoder      = encoder.to(device)
        residual_mlp = residual_mlp.to(device)
        future_encoder = None
        cfg["history_encoder_type"] = history_encoder_type
        log.info(f"Using HISTORY encoder (type={history_encoder_type}).")
        log.info(encoder)
    log.info(residual_mlp)

    residual_rollout = ResidualRollout(
        residual_mlp       = residual_mlp,
        base_model         = base_model,
        Tp                 = Tp,
        integration_method = integration_method,
        compile_inner      = compile_rollout,
    ).to(device)

    trainable_params = list(residual_mlp.parameters())
    if future_encoder is not None:
        trainable_params += list(future_encoder.parameters())
    elif encoder is not None:
        trainable_params += list(encoder.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100,
    )

    loss_fn_module = nn.L1Loss() if loss_fn == "l1" else nn.MSELoss()
    hist_prep = HistoryPreprocessor().to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    va_total = float("inf")
    uses_kl_regularization = (
        (use_future_encoder and future_encoder_type == "cvae")
        or (not use_future_encoder and history_encoder_type == "cvae")
    )

    for epoch in range(1, n_epochs + 1):
        kl_weight = (
            kl_beta * min(1.0, epoch / max(kl_warmup_epochs, 1))
            if uses_kl_regularization else 0.0
        )

        # -- train --
        if encoder is not None:
            encoder.train(True)
        residual_rollout.residual_mlp.train(True)
        if future_encoder is not None:
            future_encoder.train(True)
        tr_total_acc = tr_recon_acc = tr_kl_acc = 0.0
        n_train = 0
        for batch in train_loader:
            M, U, X0, X, *_ = batch
            if future_encoder is not None:
                U_future = U if condition_on_controls else None
                X_nom    = compute_nominal_rollout(
                    base_model, X0.squeeze(1), U, Tp, integration_method
                ) if condition_on_nominal_rollout else None
                total, recon, kl = rollout_free_running_loss_with_future_encoder(
                    future_encoder=future_encoder, residual_rollout=residual_rollout,
                    U=U, X0=X0, X=X,
                    kl_weight=kl_weight, loss_fn=loss_fn_module,
                    chunk_size=chunk_size,
                    U_future=U_future, X_nom=X_nom,
                )
            else:
                total, recon, kl = rollout_free_running_loss(
                    encoder=encoder, residual_rollout=residual_rollout,
                    M=M, U=U, X0=X0, X=X,
                    kl_weight=kl_weight, loss_fn=loss_fn_module,
                    hist_prep=hist_prep, chunk_size=chunk_size,
                )
            optimizer.zero_grad()
            # ── Lipschitz regularisation ──────────────────────────────────
            if lipschitz_lambda > 0.0:
                n_lip = min(X0.shape[0], lipschitz_n_samples)
                z_r   = torch.randn(n_lip, z_dim, device=device)
                x_r   = X0.squeeze(1)[:n_lip].detach()
                u_r   = U[:n_lip, 0].detach()
                t_r   = torch.zeros(1, device=device)
                lip_sq = estimate_lipschitz_wrt_z(
                    residual_rollout.residual_mlp, t_r, x_r, u_r, z_r,
                    create_graph=True,
                )
                total = total + lipschitz_lambda * lip_sq
            total.backward()
            nn.utils.clip_grad_norm_(
                trainable_params,
                grad_clip,
            )
            optimizer.step()
            tr_total_acc += total.item(); tr_recon_acc += recon.item(); tr_kl_acc += kl.item()
            n_train += 1
        tr_total = tr_total_acc / max(n_train, 1)
        tr_recon = tr_recon_acc / max(n_train, 1)
        tr_kl    = tr_kl_acc    / max(n_train, 1)

        # -- val --
        if encoder is not None:
            encoder.train(False)
        residual_rollout.residual_mlp.train(False)
        if future_encoder is not None:
            future_encoder.train(False)
        va_total_acc = va_recon_acc = va_kl_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                M, U, X0, X, *_ = batch
                if future_encoder is not None:
                    U_future = U if condition_on_controls else None
                    X_nom    = compute_nominal_rollout(
                        base_model, X0.squeeze(1), U, Tp, integration_method
                    ) if condition_on_nominal_rollout else None
                    total, recon, kl = rollout_free_running_loss_with_future_encoder(
                        future_encoder=future_encoder, residual_rollout=residual_rollout,
                        U=U, X0=X0, X=X,
                        kl_weight=kl_weight, loss_fn=loss_fn_module,
                        chunk_size=chunk_size,
                        U_future=U_future, X_nom=X_nom,
                    )
                else:
                    total, recon, kl = rollout_free_running_loss(
                        encoder=encoder, residual_rollout=residual_rollout,
                        M=M, U=U, X0=X0, X=X,
                        kl_weight=kl_weight, loss_fn=loss_fn_module,
                        hist_prep=hist_prep, chunk_size=chunk_size,
                    )
                va_total_acc += total.item(); va_recon_acc += recon.item(); va_kl_acc += kl.item()
                n_val += 1
        va_total = va_total_acc / max(n_val, 1)
        va_recon = va_recon_acc / max(n_val, 1)
        va_kl    = va_kl_acc    / max(n_val, 1)

        # ── Lipschitz monitoring (no graph needed) ────────────────────────
        lip_frob = None
        if epoch % lipschitz_log_freq == 0 or epoch == 1:
            with torch.no_grad():
                # use last val batch for a representative estimate
                n_lip = min(X0.shape[0], lipschitz_n_samples)
                z_r   = torch.randn(n_lip, z_dim, device=device)
                x_r   = X0.squeeze(1)[:n_lip]
                u_r   = U[:n_lip, 0]
                t_r   = torch.zeros(1, device=device)
            # estimate_lipschitz_wrt_z uses its own autograd graph; no_grad
            # must be off so torch.autograd.grad can run
            lip_sq   = estimate_lipschitz_wrt_z(
                residual_rollout.residual_mlp, t_r, x_r, u_r, z_r,
                create_graph=False,
            )
            lip_frob = lip_sq.sqrt().item()

        lip_str = f"  lip_frob {lip_frob:.4f}" if lip_frob is not None else ""
        reg_label = "kl" if uses_kl_regularization else "latent_reg"
        log.info(
            f"Epoch {epoch:4d}/{n_epochs}  "
            f"train [{tr_total:.6f} = {tr_recon:.6f} recon + {kl_weight:.4f}*{tr_kl:.6f} {reg_label}]  "
            f"val [{va_total:.6f} = {va_recon:.6f} recon + {kl_weight:.4f}*{va_kl:.6f} {reg_label}]"
            f"{lip_str}"
        )

        log_dict = dict(
            epoch=epoch, kl_weight=kl_weight,
            train_total=tr_total, train_recon=tr_recon, train_kl=tr_kl,
            val_total=va_total,   val_recon=va_recon,   val_kl=va_kl,
            lr=scheduler.get_last_lr()[0],
        )
        if not uses_kl_regularization:
            log_dict["train_latent_reg"] = tr_kl
            log_dict["val_latent_reg"] = va_kl
        if lip_frob is not None:
            log_dict["lip_frob_z"] = lip_frob
        wandb.log(log_dict)

        if va_total < best_val_loss:
            best_val_loss = va_total
            torch.save(
                {
                    "epoch":          epoch,
                    "encoder":        deepcopy(encoder.state_dict()) if encoder is not None else None,
                    "future_encoder": deepcopy(future_encoder.state_dict()) if future_encoder is not None else None,
                    "residual_mlp":   deepcopy(residual_mlp.state_dict()),
                    "optimizer":      optimizer.state_dict(),
                    "cfg":            cfg,
                    "val_loss":       va_total,
                },
                save_path / "best_residual_model.pt",
            )
            log.info(f"  new best saved  (val {va_total:.6f})")

    torch.save(
        {
            "epoch":          n_epochs,
            "encoder":        encoder.state_dict() if encoder is not None else None,
            "future_encoder": future_encoder.state_dict() if future_encoder is not None else None,
            "residual_mlp":   residual_mlp.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "cfg":            cfg,
            "val_loss":       va_total,
        },
        save_path / "last_residual_model.pt",
    )

    log.info(f"Training finished.  Best val loss: {best_val_loss:.6f}")
    run.finish()
    return best_val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment(experiment)
