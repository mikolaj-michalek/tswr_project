"""
Dataset support analysis
========================
For each wheel (front / rear) places a regular grid in the
(slip_ratio [−], slip_angle [deg]) plane and, at every grid node,
computes the **average distance** to the N closest dataset samples that
each come from a *different* run.

High value  →  region is poorly represented across runs (low support).
Low value   →  region is consistently visited across many runs (high support).

Distances are computed in σ-normalised space (each feature scaled by its
own standard deviation over the full dataset) so that slip ratio and slip
angle contribute equally regardless of their raw numerical ranges.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from scipy.spatial import KDTree
from scipy.ndimage import median_filter, grey_closing

# ── Vehicle parameters ────────────────────────────────────────────────────────
L   = 0.33    # wheelbase [m]
lr  = 0.145   # CoG → rear axle [m]
lf  = L - lr  # CoG → front axle [m]
eps = 1e-6    # numerical stability

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH = "datasets/f1tenth/260130_mpc_expert_train.csv"
df = pd.read_csv(DATASET_PATH)

v_x   = df["v_x"].to_numpy()
v_y   = df["v_y"].to_numpy()
r     = df["r"].to_numpy()
delta = df["delta"].to_numpy()

omega_front = df.get("omega_wheels_front", df["omega_wheels"]).to_numpy()
omega_rear  = df.get("omega_wheels_rear",  df["omega_wheels"]).to_numpy()
run_id = (df["run_id"].to_numpy() if "run_id" in df.columns
          else np.zeros(len(df), dtype=int))

# ── Slip quantities ───────────────────────────────────────────────────────────
slip_angle_front = np.arctan((v_y + lf * r) / (v_x + eps)) - delta
slip_angle_rear  = np.arctan((v_y - lr * r) / (v_x + eps))

v_front = v_x * np.cos(delta) + (v_y + r * lf) * np.sin(delta)

slip_ratio_front = (omega_front - v_front) / (np.maximum(omega_front, v_front) + eps)
slip_ratio_rear  = (omega_rear  - v_x)     / (np.maximum(omega_rear,  v_x)     + eps)

unique_runs  = np.unique(run_id)
n_runs_total = len(unique_runs)
print(f"Dataset : {DATASET_PATH}")
print(f"Samples : {len(df)},  distinct runs : {n_runs_total}")

# ── Analysis parameters ───────────────────────────────────────────────────────
N_REQUIRED         = min(7, n_runs_total)  # number of distinct runs per grid node
GRID_RES           = 100               # grid points along each axis (GRID_RES²)
PERCENTILE_LO      = 0.5              # lower data-range clip percentile
PERCENTILE_HI      = 99.5             # upper data-range clip percentile
GRID_MARGIN        = 0.15             # fractional margin beyond the clip range
#MEDIAN_FILTER_SIZE = 21                # kernel size for median filter (set to 1 to disable)
MEDIAN_FILTER_SIZE = 1                # kernel size for median filter (set to 1 to disable)
GREY_CLOSE_SIZE    = 5                # kernel size for grayscale morphological closing (set to 1 to disable)
SUPPORT_THRESHOLD  = 0.1             # σ-normalised distance threshold for binary support map


# ── Core function ─────────────────────────────────────────────────────────────
def avg_dist_n_distinct_runs(
    grid_pts: np.ndarray,
    data_pts: np.ndarray,
    run_ids:  np.ndarray,
    n_req:    int,
) -> np.ndarray:
    """
    For every row of *grid_pts*, find the *n_req* nearest rows in *data_pts*
    (L2, pre-normalised) that each belong to a **different** run, and return
    their mean distance.

    Parameters
    ----------
    grid_pts : (G, 2) float  – query grid points (normalised)
    data_pts : (D, 2) float  – dataset points    (normalised)
    run_ids  : (D,)          – run identifier per data point
    n_req    : int           – required number of distinct runs

    Returns
    -------
    avg_dist : (G,) float    – NaN where < n_req distinct runs were found
    """
    if n_runs_total < n_req:
        raise ValueError(
            f"Only {n_runs_total} distinct runs available; n_req={n_req}."
        )

    # Query enough candidates to reliably find n_req distinct runs
    k_cands = min(len(data_pts), max(200 * n_req, 2000))
    tree = KDTree(data_pts)

    # Single batched query for all grid points → (G, k_cands)
    print(f"    KDTree batch query  k={k_cands} … ", end="", flush=True)
    all_dists, all_idxs = tree.query(grid_pts, k=k_cands)
    print("done.")

    result = np.full(len(grid_pts), np.nan)
    for i in range(len(grid_pts)):
        seen: dict[int, float] = {}
        for d, idx in zip(all_dists[i], all_idxs[i]):
            rid = run_ids[idx]
            if rid not in seen:
                seen[rid] = float(d)
            if len(seen) == n_req:
                break
        if len(seen) == n_req:
            result[i] = np.mean(list(seen.values()))

    return result


# ── Per-wheel analysis & visualisation ───────────────────────────────────────
def analyse_wheel(
    ax_map:     plt.Axes,
    ax_runs:    plt.Axes,
    ax_binary:  plt.Axes,
    sr:         np.ndarray,
    sa_deg:     np.ndarray,
    run_ids:    np.ndarray,
    title:      str,
    data_color: str,
) -> None:
    """
    Full pipeline for one wheel, drawn on three axes:
      ax_map    – distance heatmap (optionally median-filtered) with data overlay
      ax_runs   – raw data scatter coloured by run (sanity check)
      ax_binary – thresholded binary support map after median filtering
    """
    data_raw = np.column_stack([sr, sa_deg])          # (N, 2)

    # ── σ-normalise so both axes contribute equally to distances ──────────────
    scale     = data_raw.std(axis=0)
    scale     = np.where(scale < 1e-12, 1.0, scale)
    data_norm = data_raw / scale

    # ── Build display grid in original (SR, SA-deg) space ────────────────────
    lo  = np.percentile(data_raw, PERCENTILE_LO, axis=0)   # [lo_sr, lo_sa]
    hi  = np.percentile(data_raw, PERCENTILE_HI, axis=0)   # [hi_sr, hi_sa]
    pad = GRID_MARGIN * (hi - lo)

    sr_vec = np.linspace(lo[0] - pad[0], hi[0] + pad[0], GRID_RES)
    sa_vec = np.linspace(lo[1] - pad[1], hi[1] + pad[1], GRID_RES)
    SR_g, SA_g = np.meshgrid(sr_vec, sa_vec)

    # Flatten → (G, 2) and normalise with the same scale
    grid_raw  = np.column_stack([SR_g.ravel(), SA_g.ravel()])
    grid_norm = grid_raw / scale

    # ── Compute support distances ─────────────────────────────────────────────
    print(f"  [{title}]  grid {GRID_RES}² = {GRID_RES**2} points")
    dist = avg_dist_n_distinct_runs(grid_norm, data_norm, run_ids, N_REQUIRED)
    DIST = dist.reshape(SR_g.shape)

    # ── Optional median filter ────────────────────────────────────────────────
    # NaN cells (outside dataset reach) are filled with a large sentinel value
    # so the filter treats them as high-distance, then restored afterwards.
    nan_mask = np.isnan(DIST)
    sentinel = np.nanmax(DIST) * 10 if not np.all(nan_mask) else 1.0
    DIST_filled = np.where(nan_mask, sentinel, DIST)
    if MEDIAN_FILTER_SIZE > 1:
        DIST_filt = median_filter(DIST_filled, size=MEDIAN_FILTER_SIZE, mode="nearest")
        filter_label = f"median k={MEDIAN_FILTER_SIZE}"
    else:
        DIST_filt = DIST_filled.copy()
        filter_label = "unfiltered"

    # ── Optional grayscale morphological closing ──────────────────────────────
    # Closing = dilation then erosion: fills small low-distance valleys/holes
    # in the distance map while preserving the overall high-distance regions.
    if GREY_CLOSE_SIZE > 1:
        DIST_filt = grey_closing(DIST_filt, size=GREY_CLOSE_SIZE)
        filter_label += f", grey-close k={GREY_CLOSE_SIZE}"

    DIST_filt[nan_mask] = np.nan          # restore NaN mask

    # ── Distance heatmap (filtered) ───────────────────────────────────────────
    flat_filt = DIST_filt[~nan_mask]
    vmin = max(np.nanpercentile(flat_filt, 2),  1e-9)
    vmax =     np.nanpercentile(flat_filt, 98)
    pcm  = ax_map.pcolormesh(
        SR_g, SA_g, DIST_filt,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="plasma",
        shading="auto",
        alpha=0.90,
        zorder=0,
    )
    cb = plt.colorbar(pcm, ax=ax_map)
    cb.set_label(
        f"Mean dist to {N_REQUIRED} diff-run NNs\n(σ-normalised, {filter_label})",
        fontsize=8,
    )

    # ── Overlay raw data scatter ──────────────────────────────────────────────
    ax_map.scatter(
        sr, sa_deg,
        s=1, alpha=0.20, c=data_color,
        rasterized=True, label="data",
        zorder=1,
    )
    ax_map.set_title(f"{title}  –  support map  [{filter_label}]")
    ax_map.set_xlabel("Slip ratio [−]")
    ax_map.set_ylabel("Slip angle [deg]")
    #ax_map.legend(loc="upper right", markerscale=6, fontsize=8)

    # ── Binary support map ────────────────────────────────────────────────────
    # 1 (supported) where filtered distance < SUPPORT_THRESHOLD, else 0.
    # NaN cells are shown in grey.
    BINARY = np.where(nan_mask, np.nan, (DIST_filt < SUPPORT_THRESHOLD).astype(float))
    cmap_bin = plt.cm.RdYlGn          # green = supported, red = not supported
    cmap_bin.set_bad(color="lightgrey")
    ax_binary.pcolormesh(
        SR_g, SA_g, BINARY,
        cmap=cmap_bin,
        vmin=0, vmax=1,
        shading="auto",
        alpha=0.90,
        zorder=0,
    )
    # Colour-bar with explicit ticks
    sm = plt.cm.ScalarMappable(cmap=cmap_bin,
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb2 = plt.colorbar(sm, ax=ax_binary, ticks=[0, 1])
    cb2.ax.set_yticklabels(["unsupported", "supported"], fontsize=8)
    ax_binary.scatter(
        sr, sa_deg,
        s=1, alpha=0.15, c=data_color,
        rasterized=True, zorder=1,
    )
    supported_frac = np.nanmean(BINARY)
    ax_binary.set_title(
        f"{title}  –  binary support\n"
        f"(thr={SUPPORT_THRESHOLD}, {filter_label},  "
        f"{100*supported_frac:.1f} % supported)"
    )
    ax_binary.set_xlabel("Slip ratio [−]")
    ax_binary.set_ylabel("Slip angle [deg]")

    # ── Sanity-check: scatter coloured by run ─────────────────────────────────
    uniq   = np.unique(run_ids)
    cmap_r = plt.get_cmap("tab10", min(len(uniq), 10))
    for k, rid in enumerate(uniq):
        m = run_ids == rid
        ax_runs.scatter(
            sr[m], sa_deg[m],
            s=1, alpha=0.30,
            color=cmap_r(k % 10),
            rasterized=True,
            label=f"run {rid}",
        )
    ax_runs.set_title(f"{title}  –  data by run")
    ax_runs.set_xlabel("Slip ratio [−]")
    ax_runs.set_ylabel("Slip angle [deg]")
    #hl, ll = ax_runs.get_legend_handles_labels()
    #ax_runs.legend(
    #    hl, ll,
    #    markerscale=6, fontsize=7,
    #    loc="upper right",
    #    ncol=max(1, len(uniq) // 8),
    #)


# ── Build figure (2 rows × 3 cols) ───────────────────────────────────────────
# col 0: filtered distance heatmap   col 1: per-run scatter   col 2: binary map
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(
    f"Dataset support  –  mean dist to {N_REQUIRED} nearest points "
    f"from {N_REQUIRED} distinct runs  |  "
    f"median filter k={MEDIAN_FILTER_SIZE}  |  threshold={SUPPORT_THRESHOLD}\n"
    f"{DATASET_PATH}  ({len(df)} samples, {n_runs_total} runs)",
    fontsize=11,
)

analyse_wheel(
    axes[0, 0], axes[0, 1], axes[0, 2],
    slip_ratio_front, np.degrees(slip_angle_front),
    run_id, "Front wheel", "tab:green",
)
analyse_wheel(
    axes[1, 0], axes[1, 1], axes[1, 2],
    slip_ratio_rear, np.degrees(slip_angle_rear),
    run_id, "Rear wheel", "tab:green",
)

plt.tight_layout()
plt.savefig("experiments/imgs/dataset_support_analysis.png", dpi=150)
plt.show()
