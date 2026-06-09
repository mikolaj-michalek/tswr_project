"""
Train a neural network to predict distance-to-dataset
=====================================================
The network maps 4-D slip coordinates
    (sa_front, sr_front, sa_rear, sr_rear)
to a scalar distance value (σ-normalised, as defined in
``generate_distance_dataset.py``).

Supports ``experiment_launcher`` for hyper-parameter sweeps and ``wandb``
for logging.

Usage
-----
    python train_distance_model.py                      # defaults
    python train_distance_model.py --n_epochs 500 --lr 1e-3
"""

import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from experiment_launcher import run_experiment, single_experiment
from torch.utils.data import DataLoader, TensorDataset

from ldm.systems.commons.distance_model import DistanceModel

log = logging.getLogger(__name__)

# ── Torch defaults ────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = True

DEBUG = True
#DEBUG = False


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_distance_dataset(csv_path: str, device: str = "cpu"
                          ) -> TensorDataset:
    """Load a CSV produced by ``generate_distance_dataset.py``."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    X = torch.tensor(
        df[["sa_front", "sr_front", "sa_rear", "sr_rear"]].values,
        dtype=torch.float32, device=device,
    )
    Y = torch.tensor(
        df["distance"].values, dtype=torch.float32, device=device,
    ).unsqueeze(-1)
    return TensorDataset(X, Y)


def make_loader(dataset: TensorDataset, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_normalisation(dataset: TensorDataset):
    """Return (input_mean, input_std, target_mean, target_std) from a dataset."""
    X, Y = dataset.tensors
    x_mean, x_std = X.mean(0), X.std(0).clamp(min=1e-8)
    y_mean, y_std = Y.mean(0), Y.std(0).clamp(min=1e-8)
    return x_mean, x_std, y_mean, y_std


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

@single_experiment
def experiment(
    # ── Data ──────────────────────────────────────────────────────────────
    # ── Architecture ──────────────────────────────────────────────────────
    #hidden_sizes:  str   = "16,16",   # comma-separated
    hidden_sizes:  str   = "128,128",   # comma-separated
    activation:    str   = "silu",
    #activation:    str   = "relu",
    #activation:    str   = "tanh",
    # ── Optimisation ──────────────────────────────────────────────────────
    n_epochs:      int   = 300,
    batch_size:    int   = 512,
    lr:            float = 1e-3,
    #weight_decay:  float = 1e-5,
    weight_decay:  float = 0.,
    grad_clip:     float = 1.0,
    #scheduler:     str   = "cosine",        # "cosine" | "none"
    scheduler:     str   = "none",        # "cosine" | "none"
    # ── Misc ──────────────────────────────────────────────────────────────
    normalise_io:  bool  = True,
    compile:       bool  = True,
    results_dir:   str   = "./results",
    seed:          int   = 0,
):
    config = {**locals()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    hidden = [int(s) for s in hidden_sizes.split(",")]

    log.info(f"PID {os.getpid()}  seed {seed}  device {device}")

    # ── W&B ───────────────────────────────────────────────────────────────
    group_name = (f"dist_mlp_{'x'.join(map(str, hidden))}_{activation}"
                  f"_lr{lr}_bs{batch_size}_wd{weight_decay}{'_nonorm' if not normalise_io else ''}"
                  f"{scheduler if scheduler != 'none' else ''}")
    run = wandb.init(
        project="distance_model",
        group=group_name,
        name=f"{group_name}_seed{seed}",
        config=config,
        mode="disabled" if DEBUG else "online",
    )

    model_save_dir = Path("./results") / "distance_models" / run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = Path(os.path.join(os.path.dirname(__file__), "..",
                                 "datasets/f1tenth/distance_datasets"))
    train_ds = load_distance_dataset(data_dir / "distance_train.csv", device)
    val_ds   = load_distance_dataset(data_dir / "distance_val.csv",   device)

    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False)

    log.info(f"Train: {len(train_ds)} samples,  Val: {len(val_ds)} samples")

    # ── Normalisation ─────────────────────────────────────────────────────
    x_mean, x_std, y_mean, y_std = compute_normalisation(train_ds)
    if not normalise_io:
        x_mean, x_std = torch.zeros_like(x_mean), torch.ones_like(x_std)
        y_mean, y_std = torch.zeros_like(y_mean), torch.ones_like(y_std)

    # ── Model ─────────────────────────────────────────────────────────────
    model = DistanceModel(
        hidden_sizes=hidden,
        activation=activation,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
    ).to(device)
    log.info(f"Model:\n{model}")
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    if compile:
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    if scheduler == "cosine":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=n_epochs, eta_min=lr * 0.01)
    else:
        lr_sched = None

    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        t0 = time.perf_counter()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)

            optimiser.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

            train_losses.append(loss.item())

        if lr_sched is not None:
            lr_sched.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                Y_pred = model(X_batch)
                val_losses.append(loss_fn(Y_pred, Y_batch).item())

        t1 = time.perf_counter()

        mean_train = np.mean(train_losses)
        mean_val   = np.mean(val_losses)
        current_lr = optimiser.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────────────
        log_dict = {
            "train_loss": mean_train,
            "val_loss":   mean_val,
            "best_val":   best_val_loss,
            "lr":         current_lr,
            "epoch_time": t1 - t0,
        }
        wandb.log(log_dict, step=epoch)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            log.info(f"ep {epoch:4d}/{n_epochs}  "
                     f"train {mean_train:.6f}  val {mean_val:.6f}  "
                     f"best {best_val_loss:.6f}  lr {current_lr:.2e}  "
                     f"{t1-t0:.2f}s")

        # ── Checkpoint ────────────────────────────────────────────────────
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            # Unwrap compiled model to get clean state_dict keys
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = deepcopy(raw_model.state_dict())

        if best_state is not None and (
            (epoch + 1) % 50 == 0 or epoch == n_epochs - 1
        ):
            ckpt_path = model_save_dir / f"distance_model_ep{epoch + 1}.pt"
            torch.save({
                "model_state":  best_state,
                "hidden_sizes": hidden,
                "activation":   activation,
                "config":       config,
                "epoch":        epoch + 1,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            log.info(f"Checkpoint saved → {ckpt_path}  (val loss {best_val_loss:.6f})")

        if torch.isnan(torch.tensor([mean_train, mean_val])).any():
            log.error("NaN detected – stopping.")
            break

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = model_save_dir / f"distance_model_ep{n_epochs}.pt"
    log.info(f"Best model → {save_path}  (val loss {best_val_loss:.6f})")

    #artifact = wandb.Artifact("distance_model", type="model")
    #artifact.add_dir(str(out_dir))
    #artifact.save()
    run.finish()

    return best_val_loss


if __name__ == "__main__":
    run_experiment(experiment)
