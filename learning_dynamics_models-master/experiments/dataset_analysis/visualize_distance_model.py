"""
Visualise a trained distance-to-dataset model
==============================================
Produces 2-D slice plots of the learned distance function alongside the
underlying source dataset and the distance training data.

The 4-D slip space (sa_front, sr_front, sa_rear, sr_rear) is shown via
six canonical 2-D projections (one per pair of axes).  For each projection
the remaining two coordinates are held at their dataset-median values.

Additionally, four "marginal" 1-D plots show the predicted distance as a
function of each axis individually (others at median).

Usage
-----
    python visualize_distance_model.py --model_path results/distance_models/<run>/distance_model.pt
    python visualize_distance_model.py --model_path <path> --source_dataset <csv>
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm

from ldm.systems.commons.distance_model import DistanceModel

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "../results/distance_models",
                          "dist_mlp_128x128_silu_lr0.001_bs512_wd0.0_seed0",
                          #"dist_mlp_16x16_relu_lr0.001_bs512_wd1e-05_seed1",
                          #"dist_mlp_16x16_tanh_lr0.001_bs512_wd0.0_seed1",
                          "distance_model_ep300.pt")
SOURCE_DATASET    = "datasets/f1tenth/260130_mpc_expert_train.csv"
DIST_DATA_DIR     = "datasets/f1tenth/distance_datasets"
OUTPUT_DIR        = "experiments/imgs"
GRID_RES          = 120
#PERCENTILE_MARGIN = 5.0
PERCENTILE_MARGIN = 1.0

# ── Re-use slip computation from the generator ───────────────────────────────
# Inline a minimal version so this script is self-contained.

@dataclass
class VehicleParams:
    wheelbase: float = 0.33
    lr:        float = 0.145
    @property
    def lf(self) -> float:
        return self.wheelbase - self.lr


def compute_slips(df: pd.DataFrame, vp: VehicleParams, eps: float = 1e-6):
    v_x   = df["v_x"].to_numpy()
    v_y   = df["v_y"].to_numpy()
    r     = df["r"].to_numpy()
    delta = df["delta"].to_numpy()
    omega_f = df.get("omega_wheels_front", df["omega_wheels"]).to_numpy()
    omega_r = df.get("omega_wheels_rear",  df["omega_wheels"]).to_numpy()

    sa_f = np.arctan((v_y + vp.lf * r) / (v_x + eps)) - delta
    sa_r = np.arctan((v_y - vp.lr * r) / (v_x + eps))
    v_f  = v_x * np.cos(delta) + (v_y + r * vp.lf) * np.sin(delta)
    sr_f = (omega_f - v_f)  / (np.maximum(omega_f, v_f)  + eps)
    sr_r = (omega_r - v_x)  / (np.maximum(omega_r, v_x)  + eps)
    return sa_f, sr_f, sa_r, sr_r


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(dist_model: DistanceModel, x_raw: np.ndarray) -> np.ndarray:
    """Run the model on raw (un-normalised) 4-D input and return predicted distance."""
    X = torch.tensor(x_raw, dtype=torch.float32)
    return dist_model(X).squeeze(-1).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

DIM_NAMES = ["SA front [rad]", "SR front [−]", "SA rear [rad]", "SR rear [−]"]
DIM_KEYS  = ["sa_front", "sr_front", "sa_rear", "sr_rear"]


def _grid_2d(ax_i: int, ax_j: int, medians: np.ndarray,
             lo: np.ndarray, hi: np.ndarray, res: int = 120):
    """Build a (res², 4) array scanning dims ax_i and ax_j, others at median."""
    vi = np.linspace(lo[ax_i], hi[ax_i], res)
    vj = np.linspace(lo[ax_j], hi[ax_j], res)
    Gi, Gj = np.meshgrid(vi, vj)
    pts = np.tile(medians, (res * res, 1))
    pts[:, ax_i] = Gi.ravel()
    pts[:, ax_j] = Gj.ravel()
    return Gi, Gj, pts


def plot_2d_slices(dist_model: DistanceModel,
                   source_4d: np.ndarray | None,
                   dist_train: pd.DataFrame | None,
                   lo: np.ndarray, hi: np.ndarray,
                   medians: np.ndarray,
                   res: int = 120,
                   output_path: str | None = None):
    """Six 2-D heatmaps – one per pair of the four axes."""
    pairs = list(combinations(range(4), 2))   # 6 pairs
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes_flat = axes.ravel()

    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]
        Gi, Gj, pts = _grid_2d(i, j, medians, lo, hi, res)
        Z = predict(dist_model, pts).reshape(Gi.shape)
        Z = np.clip(Z, 1e-6, None)  # for LogNorm

        pcm = ax.pcolormesh(Gi, Gj, Z,
                            norm=LogNorm(vmin=max(Z.min(), 1e-4), vmax=Z.max()),
                            cmap="plasma", shading="auto", alpha=0.9)
        plt.colorbar(pcm, ax=ax, label="pred distance")

        # Overlay source dataset points
        if source_4d is not None:
            ax.scatter(source_4d[:, i], source_4d[:, j],
                       s=0.4, alpha=0.15, c="lime", rasterized=True,
                       label="source data")

        # Overlay distance-dataset training points
        if dist_train is not None:
            ax.scatter(dist_train[DIM_KEYS[i]], dist_train[DIM_KEYS[j]],
                       s=0.8, alpha=0.10, c="cyan", rasterized=True,
                       label="dist train")

        ax.set_xlabel(DIM_NAMES[i])
        ax.set_ylabel(DIM_NAMES[j])
        ax.set_title(f"{DIM_NAMES[i]}  vs  {DIM_NAMES[j]}")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   markerscale=8, fontsize=9)

    fig.suptitle("Predicted distance to dataset  –  2-D slices (others @ median)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    plt.show()


def plot_1d_marginals(dist_model: DistanceModel,
                      source_4d: np.ndarray | None,
                      lo: np.ndarray, hi: np.ndarray,
                      medians: np.ndarray,
                      res: int = 300,
                      output_path: str | None = None):
    """Four 1-D curves – predicted distance along each axis (others at median)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for dim in range(4):
        ax = axes[dim]
        vals = np.linspace(lo[dim], hi[dim], res)
        pts = np.tile(medians, (res, 1))
        pts[:, dim] = vals
        Z = predict(dist_model, pts)

        ax.plot(vals, Z, "b-", lw=1.5, label="NN prediction")
        ax.set_xlabel(DIM_NAMES[dim])
        ax.set_ylabel("distance")
        ax.set_title(f"Marginal: {DIM_NAMES[dim]}")

        # Show source data histogram (normalised) in background
        if source_4d is not None:
            ax2 = ax.twinx()
            ax2.hist(source_4d[:, dim], bins=60, alpha=0.25, color="green",
                     density=True, label="data density")
            ax2.set_ylabel("density", color="green", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="green", labelsize=7)

        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("1-D marginal distance predictions  (other dims @ median)",
                 fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    plt.show()


def plot_prediction_vs_ground_truth(dist_model: DistanceModel,
                                    test_csv: str,
                                    output_path: str | None = None):
    """Scatter of NN-predicted vs ground-truth distance on the test set."""
    df = pd.read_csv(test_csv)
    X_raw = df[DIM_KEYS].values
    y_true = df["distance"].values
    y_pred = predict(dist_model, X_raw)

    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    r2  = 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - y_true.mean())**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Scatter
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=1, alpha=0.3, rasterized=True)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1, label="ideal")
    ax.set_xlabel("Ground truth distance")
    ax.set_ylabel("Predicted distance")
    ax.set_title(f"Pred vs True  (MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f})")
    ax.legend()
    ax.set_aspect("equal", "box")

    # 2) Error histogram
    ax = axes[1]
    errors = y_pred - y_true
    ax.hist(errors, bins=80, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Prediction error (pred − true)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error distribution  (mean={errors.mean():.4f}, "
                 f"std={errors.std():.4f})")

    # 3) Error vs true distance
    ax = axes[2]
    ax.scatter(y_true, np.abs(errors), s=1, alpha=0.3, rasterized=True)
    ax.set_xlabel("Ground truth distance")
    ax.set_ylabel("|Error|")
    ax.set_title("Absolute error vs distance")

    fig.suptitle(f"Test-set evaluation  ({len(df)} samples)", fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    plt.show()


def plot_front_rear_wheel_views(dist_model: DistanceModel,
                                source_4d: np.ndarray | None,
                                lo: np.ndarray, hi: np.ndarray,
                                medians: np.ndarray,
                                res: int = 120,
                                output_path: str | None = None):
    """
    Two "wheel-centric" heatmaps resembling the original support analysis:
      - Front wheel: SA_front vs SR_front  (rear at median)
      - Rear wheel:  SA_rear  vs SR_rear   (front at median)
    """
    wheel_pairs = [(0, 1, "Front wheel"), (2, 3, "Rear wheel")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (i, j, title) in zip(axes, wheel_pairs):
        Gi, Gj, pts = _grid_2d(i, j, medians, lo, hi, res)
        Z = predict(dist_model, pts).reshape(Gi.shape)
        pcm = ax.pcolormesh(Gi, Gj, Z,
                            norm=LogNorm(vmin=max(Z.min(), 1e-4), vmax=Z.max()),
                            cmap="plasma", shading="auto", alpha=0.9)
        plt.colorbar(pcm, ax=ax, label="pred distance")

        if source_4d is not None:
            ax.scatter(source_4d[:, i], source_4d[:, j],
                       s=0.5, alpha=0.15, c="lime", rasterized=True,
                       label="source data")

        ax.set_xlabel(DIM_NAMES[i])
        ax.set_ylabel(DIM_NAMES[j])
        ax.set_title(f"{title}  –  NN distance prediction")
        ax.legend(markerscale=8, fontsize=8, loc="upper right")

    fig.suptitle("Wheel-centric distance maps  (other wheel dims @ median)",
                 fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = Path(DIST_DATA_DIR)

    # ── Load model ────────────────────────────────────────────────────────
    dist_model = DistanceModel.from_checkpoint(MODEL_PATH)
    cfg = torch.load(MODEL_PATH, map_location="cpu",
                     weights_only=False).get("config", {})
    print(f"  Config: {cfg}")

    # ── Load source dataset (vehicle data → 4-D slips) ───────────────────
    source_4d = None
    if Path(SOURCE_DATASET).exists():
        print(f"Loading source dataset: {SOURCE_DATASET}")
        df_src = pd.read_csv(SOURCE_DATASET)
        vp = VehicleParams()
        sa_f, sr_f, sa_r, sr_r = compute_slips(df_src, vp)
        source_4d = np.column_stack([sa_f, sr_f, sa_r, sr_r])
        print(f"  Source data: {source_4d.shape[0]} samples")
    else:
        print(f"  ⚠ Source dataset not found: {SOURCE_DATASET}")

    # ── Load distance training data ───────────────────────────────────────
    dist_train_path = dist_dir / "distance_train.csv"
    dist_train = pd.read_csv(dist_train_path) if dist_train_path.exists() else None

    # ── Compute axis ranges & medians ─────────────────────────────────────
    margin = PERCENTILE_MARGIN
    if source_4d is not None:
        lo = np.percentile(source_4d, margin, axis=0)
        hi = np.percentile(source_4d, 100 - margin, axis=0)
        medians = np.median(source_4d, axis=0)
    elif dist_train is not None:
        vals = dist_train[DIM_KEYS].values
        lo = np.percentile(vals, margin, axis=0)
        hi = np.percentile(vals, 100 - margin, axis=0)
        medians = np.median(vals, axis=0)
    else:
        lo = -np.ones(4)
        hi = np.ones(4)
        medians = np.zeros(4)

    # Expand range slightly
    pad = 0.25 * (hi - lo)
    lo -= pad
    hi += pad

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n── Generating plots ─────────────────────────────────")

    plot_front_rear_wheel_views(
        dist_model,
        source_4d, lo, hi, medians,
        res=GRID_RES,
        output_path=out_dir / "dist_model_wheel_views.png",
    )

    plot_2d_slices(
        dist_model,
        source_4d, dist_train, lo, hi, medians,
        res=GRID_RES,
        output_path=out_dir / "dist_model_2d_slices.png",
    )

    #plot_1d_marginals(
    #    net, x_mean, x_std, y_mean, y_std,
    #    source_4d, lo, hi, medians,
    #    res=300,
    #    output_path=out_dir / "dist_model_1d_marginals.png",
    #)

    # Test-set evaluation
    test_csv = dist_dir / "distance_test.csv"
    if test_csv.exists():
        plot_prediction_vs_ground_truth(
            dist_model,
            str(test_csv),
            output_path=out_dir / "dist_model_test_eval.png",
        )
    else:
        print(f"  ⚠ Test set not found: {test_csv}")

    print("\n✓ All visualisations complete.")


if __name__ == "__main__":
    main()
