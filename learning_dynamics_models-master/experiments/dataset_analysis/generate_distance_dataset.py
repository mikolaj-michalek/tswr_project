"""
Generate distance-to-dataset datasets for neural network training
=================================================================
Computes the "distance to dataset" in 4-D slip space:

    (slip_angle_front, slip_ratio_front, slip_angle_rear, slip_ratio_rear)

For each query point the metric is the mean σ-normalised L2 distance to the
N nearest dataset samples that each come from a *different* run (identical to
the definition in ``dataset_support_analysis.py``).

Two sources of query points are combined (50 / 50):
  1. **Perturbed real data** – each sample in the source dataset is randomly
     jittered in σ-normalised space, then mapped back to raw slip values.
  2. **Uniform random**     – points drawn uniformly in the padded bounding
     box of the source data.

Three independent datasets (train / val / test) are produced with different
random seeds so there is no data leakage.

Usage
-----
    python generate_distance_dataset.py          # uses defaults
    python generate_distance_dataset.py --n_samples 200000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleParams:
    """Geometric vehicle parameters."""
    wheelbase: float = 0.33       # L  [m]
    lr:        float = 0.145      # CoG → rear axle  [m]

    @property
    def lf(self) -> float:
        return self.wheelbase - self.lr


@dataclass
class AnalysisConfig:
    """Parameters that control the distance computation and data generation."""
    # Distance computation
    n_required:         int   = 7       # distinct runs per query point
    eps:                float = 1e-6    # numerical stability constant

    # Data range
    percentile_lo:      float = 0.5     # lower clip percentile
    percentile_hi:      float = 99.5    # upper clip percentile
    grid_margin:        float = 0.25    # fractional margin beyond clip range

    # Perturbation
    perturb_sigma:      float = 0.3     # std of Gaussian noise in σ-normalised space

    # Generated dataset sizes
    n_samples_train:    int   = 100_000
    n_samples_val:      int   = 20_000
    n_samples_test:     int   = 20_000

    # Reproducibility
    seed_train:         int   = 42
    seed_val:           int   = 123
    seed_test:          int   = 789

    # I/O
    dataset_path: str = "datasets/f1tenth/260130_mpc_expert_train.csv"
    output_dir:   str = "datasets/f1tenth/distance_datasets"


# ═══════════════════════════════════════════════════════════════════════════════
# Slip computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_slips(df: pd.DataFrame, vp: VehicleParams, eps: float = 1e-6
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute front/rear slip angles and slip ratios from a dataframe.

    Returns
    -------
    slip_angle_front, slip_ratio_front, slip_angle_rear, slip_ratio_rear
        Each of shape (N,).
    """
    v_x   = df["v_x"].to_numpy()
    v_y   = df["v_y"].to_numpy()
    r     = df["r"].to_numpy()
    delta = df["delta"].to_numpy()

    omega_front = df.get("omega_wheels_front", df["omega_wheels"]).to_numpy()
    omega_rear  = df.get("omega_wheels_rear",  df["omega_wheels"]).to_numpy()

    # Slip angles
    sa_front = np.arctan((v_y + vp.lf * r) / (v_x + eps)) - delta
    sa_rear  = np.arctan((v_y - vp.lr * r) / (v_x + eps))

    # Slip ratios
    v_front  = v_x * np.cos(delta) + (v_y + r * vp.lf) * np.sin(delta)
    sr_front = (omega_front - v_front) / (np.maximum(omega_front, v_front) + eps)
    sr_rear  = (omega_rear  - v_x)     / (np.maximum(omega_rear,  v_x)     + eps)

    return sa_front, sr_front, sa_rear, sr_rear


# ═══════════════════════════════════════════════════════════════════════════════
# Distance-to-dataset computation (4-D, σ-normalised)
# ═══════════════════════════════════════════════════════════════════════════════

def build_normalisation(data_4d: np.ndarray) -> np.ndarray:
    """
    Compute per-feature standard deviations for σ-normalisation.

    Returns
    -------
    scale : (4,) array – clamped to avoid division by zero.
    """
    scale = data_4d.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return scale


def mean_distance_to_n_distinct_runs(
    query_pts_norm: np.ndarray,
    data_pts_norm:  np.ndarray,
    run_ids:        np.ndarray,
    n_required:     int,
) -> np.ndarray:
    """
    For every query point, find the *n_required* nearest data points (L2 in
    σ-normalised space) that each belong to a **different** run, and return
    their mean distance.

    Parameters
    ----------
    query_pts_norm : (Q, 4) – query points (already σ-normalised)
    data_pts_norm  : (D, 4) – dataset points (already σ-normalised)
    run_ids        : (D,)   – run identifier per dataset point
    n_required     : int    – required number of distinct runs

    Returns
    -------
    distances : (Q,) – mean distance; NaN where < n_required runs found.
    """
    n_runs_total = len(np.unique(run_ids))
    if n_runs_total < n_required:
        raise ValueError(
            f"Only {n_runs_total} distinct runs; need n_required={n_required}."
        )

    k_cands = min(len(data_pts_norm), max(200 * n_required, 2000))
    tree = KDTree(data_pts_norm)

    print(f"    KDTree query  Q={len(query_pts_norm)}, k={k_cands} … ",
          end="", flush=True)
    all_dists, all_idxs = tree.query(query_pts_norm, k=k_cands)
    print("done.")

    result = np.full(len(query_pts_norm), np.nan)
    for i in range(len(query_pts_norm)):
        seen: dict[int, float] = {}
        for d, idx in zip(all_dists[i], all_idxs[i]):
            rid = int(run_ids[idx])
            if rid not in seen:
                seen[rid] = float(d)
            if len(seen) == n_required:
                break
        if len(seen) == n_required:
            result[i] = np.mean(list(seen.values()))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Query-point generation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bounding_box(
    data_4d: np.ndarray,
    percentile_lo: float,
    percentile_hi: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the padded bounding box of the data in raw 4-D slip space.

    Returns
    -------
    lo, hi : (4,) arrays – lower and upper bounds per dimension.
    """
    lo = np.percentile(data_4d, percentile_lo, axis=0)
    hi = np.percentile(data_4d, percentile_hi, axis=0)
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


def generate_perturbed_points(
    data_4d: np.ndarray,
    scale:   np.ndarray,
    n_points: int,
    sigma:   float,
    rng:     np.random.Generator,
) -> np.ndarray:
    """
    Sample *n_points* from the dataset (with replacement) and add
    Gaussian noise in σ-normalised space, then map back to raw space.

    Returns
    -------
    perturbed : (n_points, 4) – raw (un-normalised) 4-D slip values.
    """
    indices = rng.integers(0, len(data_4d), size=n_points)
    base = data_4d[indices]                               # (n, 4)
    noise = rng.normal(scale=sigma, size=(n_points, 4))   # in σ-normalised units
    return base + noise * scale                            # back to raw units


def generate_uniform_random_points(
    lo:       np.ndarray,
    hi:       np.ndarray,
    n_points: int,
    rng:      np.random.Generator,
) -> np.ndarray:
    """
    Draw *n_points* uniformly inside the padded bounding box.

    Returns
    -------
    random_pts : (n_points, 4) – raw 4-D slip values.
    """
    return rng.uniform(lo, hi, size=(n_points, 4))


def generate_query_points(
    data_4d:  np.ndarray,
    scale:    np.ndarray,
    lo:       np.ndarray,
    hi:       np.ndarray,
    n_total:  int,
    sigma:    float,
    rng:      np.random.Generator,
) -> np.ndarray:
    """
    Generate *n_total* query points: 50 % perturbed real data,
    50 % uniform random inside the padded bounding box.

    Returns
    -------
    query_pts : (n_total, 4) – raw slip values.
    """
    n_perturbed = n_total // 2
    n_random    = n_total - n_perturbed

    pts_perturbed = generate_perturbed_points(data_4d, scale, n_perturbed,
                                              sigma, rng)
    pts_random    = generate_uniform_random_points(lo, hi, n_random, rng)

    combined = np.concatenate([pts_perturbed, pts_random], axis=0)
    rng.shuffle(combined)                                 # mix the two sources
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# Single-split generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_split(
    split_name:     str,
    n_samples:      int,
    seed:           int,
    data_4d:        np.ndarray,
    scale:          np.ndarray,
    lo:             np.ndarray,
    hi:             np.ndarray,
    run_ids:        np.ndarray,
    cfg:            AnalysisConfig,
    output_dir:     Path,
) -> Path:
    """
    Generate one dataset split (train / val / test) and save to CSV.

    Columns: sa_front, sr_front, sa_rear, sr_rear, distance

    Returns
    -------
    path : Path to the saved CSV file.
    """
    print(f"\n{'─'*60}")
    print(f"  Generating '{split_name}' split  ({n_samples} samples, seed={seed})")
    print(f"{'─'*60}")

    rng = np.random.default_rng(seed)

    # ── Query points (raw) ────────────────────────────────────────────────────
    query_raw = generate_query_points(
        data_4d, scale, lo, hi, n_samples, cfg.perturb_sigma, rng,
    )

    # ── Compute distances in σ-normalised space ──────────────────────────────
    data_norm  = data_4d / scale
    query_norm = query_raw / scale

    distances = mean_distance_to_n_distinct_runs(
        query_norm, data_norm, run_ids, cfg.n_required,
    )

    # ── Drop queries where distance could not be computed (NaN) ──────────────
    valid = ~np.isnan(distances)
    n_valid = valid.sum()
    n_nan   = (~valid).sum()
    if n_nan > 0:
        print(f"    ⚠  {n_nan} / {n_samples} queries had < {cfg.n_required} "
              f"distinct-run neighbours → dropped.")
    query_raw = query_raw[valid]
    distances = distances[valid]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "sa_front":  query_raw[:, 0],
        "sr_front":  query_raw[:, 1],
        "sa_rear":   query_raw[:, 2],
        "sr_rear":   query_raw[:, 3],
        "distance":  distances,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"distance_{split_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"    ✓  Saved {n_valid} rows → {out_path}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate distance-to-dataset datasets for NN training.",
    )
    p.add_argument("--dataset", type=str, default=None,
                   help="Path to the source CSV dataset (overrides config).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for output CSVs (overrides config).")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Total samples per split (overrides individual sizes).")
    p.add_argument("--n_required", type=int, default=None,
                   help="Number of distinct runs per query (overrides config).")
    p.add_argument("--perturb_sigma", type=float, default=None,
                   help="Perturbation σ in normalised space (overrides config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = AnalysisConfig()
    vp   = VehicleParams()

    # Override config with CLI arguments
    if args.dataset is not None:
        cfg.dataset_path = args.dataset
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.n_samples is not None:
        cfg.n_samples_train = args.n_samples
        cfg.n_samples_val   = args.n_samples
        cfg.n_samples_test  = args.n_samples
    if args.n_required is not None:
        cfg.n_required = args.n_required
    if args.perturb_sigma is not None:
        cfg.perturb_sigma = args.perturb_sigma

    # ── Load source dataset ───────────────────────────────────────────────────
    print(f"Loading dataset: {cfg.dataset_path}")
    df = pd.read_csv(cfg.dataset_path)
    run_ids = (df["run_id"].to_numpy() if "run_id" in df.columns
               else np.zeros(len(df), dtype=int))
    n_runs = len(np.unique(run_ids))
    cfg.n_required = min(cfg.n_required, n_runs)
    print(f"  Samples: {len(df)},  distinct runs: {n_runs},  "
          f"n_required: {cfg.n_required}")

    # ── Compute slip quantities ───────────────────────────────────────────────
    sa_f, sr_f, sa_r, sr_r = compute_slips(df, vp, cfg.eps)
    data_4d = np.column_stack([sa_f, sr_f, sa_r, sr_r])   # (N, 4)

    # ── Normalisation & bounding box (computed once, shared across splits) ────
    scale = build_normalisation(data_4d)
    lo, hi = compute_bounding_box(
        data_4d, cfg.percentile_lo, cfg.percentile_hi, cfg.grid_margin,
    )
    print(f"  σ-scale : {scale}")
    print(f"  BBox lo : {lo}")
    print(f"  BBox hi : {hi}")

    # ── Generate splits ───────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)

    splits = [
        ("train", cfg.n_samples_train, cfg.seed_train),
        ("val",   cfg.n_samples_val,   cfg.seed_val),
        ("test",  cfg.n_samples_test,  cfg.seed_test),
    ]

    saved_paths = []
    for name, n, seed in splits:
        path = generate_split(
            name, n, seed,
            data_4d, scale, lo, hi, run_ids,
            cfg, output_dir,
        )
        saved_paths.append(path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  All splits generated successfully!")
    print(f"{'═'*60}")
    for p in saved_paths:
        n_rows = sum(1 for _ in open(p)) - 1   # minus header
        print(f"    {p.name:30s}  {n_rows:>8,} rows")
    print()


if __name__ == "__main__":
    main()
