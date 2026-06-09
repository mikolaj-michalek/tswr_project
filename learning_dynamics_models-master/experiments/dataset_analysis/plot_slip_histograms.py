import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# ── Plot selection ───────────────────────────────────────────────────────────
PLOT_STATE_VARIABLES  = False   # v_x / v_y / r over time
PLOT_PSD              = False   # per-run Welch PSD
PLOT_SLIP_HISTOGRAMS  = True   # slip angle & ratio histograms
PLOT_SR_SA_SCATTER    = True   # 2-D slip ratio vs slip angle scatter + fit

# ── Overlay (policy / simulation trace) ─────────────────────────────────────
PLOT_OVERLAY      = True    # overlay data from a second CSV on all plots
#OVERLAY_FILE = "monza_kicajka4_n10_nup7.csv"
OVERLAY_FILE = "monza_neural_tires.csv"
OVERLAY_PATH      = os.path.join(os.path.dirname(__file__), OVERLAY_FILE)
OVERLAY_LABEL     = "policy (monza)"   # legend label for the overlay
OVERLAY_COLOR     = "darkorange"       # colour used for the overlay data

# ── Distance-to-dataset evaluation ────────────────────────────────────────────
# Requires PLOT_OVERLAY = True.  Loads a trained DistanceModel, evaluates it
# on the policy slip coordinates and produces colour-coded diagnostics.
PLOT_DISTANCE_EVAL  = True
DISTANCE_MODEL_PATH = os.path.join(os.path.dirname(__file__),
                                    "../results/distance_models",
                                    "dist_mlp_128x128_silu_lr0.001_bs512_wd0.0_seed0",
                                    "distance_model_ep300.pt")
DISTANCE_THRESHOLD  = 0.35   # points with distance > threshold are flagged as OOD

# ── Vehicle parameters ────────────────────────────────────────────────────────
#L   = 2.5    # wheelbase [m]
#lr  = 1.25   # CoG → rear axle [m]
L   = 0.33    # wheelbase [m]
lr  = 0.145   # CoG → rear axle [m]
lf  = L - lr # CoG → front axle [m]
eps = 1e-6   # numerical stability

# ── Load dataset ──────────────────────────────────────────────────────────────
#dataset_path = "datasets/vehicle_dynamics/oct_6_2_2/test.csv"
dataset_path = "datasets/f1tenth/260130_mpc_expert_train.csv"
#dataset_path = "datasets/f1tenth/260130_mpc_expert_sgf_p5_w25_train.csv"
df = pd.read_csv(dataset_path)

v_x         = df["v_x"].to_numpy()
v_y         = df["v_y"].to_numpy()
r           = df["r"].to_numpy()
delta       = df["delta"].to_numpy()
omega_front = df.get("omega_wheels_front", df["omega_wheels"]).to_numpy()
omega_rear  = df.get("omega_wheels_rear", df["omega_wheels"]).to_numpy()
run_id      = df.get("run_id", np.zeros(len(df), dtype=int)).to_numpy()

unique_runs = np.unique(run_id)
cmap = plt.get_cmap("tab10", 10) 
#run_colors = {rid: cmap(i) for i, rid in enumerate(unique_runs)}

# ── Load overlay data ─────────────────────────────────────────────────────────
if PLOT_OVERLAY:
    df_ov = pd.read_csv(OVERLAY_PATH)
    ov_v_x         = df_ov["v_x"].to_numpy()
    ov_v_y         = df_ov["v_y"].to_numpy()
    ov_r           = df_ov["r"].to_numpy()
    ov_delta       = df_ov["delta"].to_numpy()
    ov_omega_front = df_ov.get("omega_wheels_front", df_ov["omega_wheels"]).to_numpy()
    ov_omega_rear  = df_ov.get("omega_wheels_rear",  df_ov["omega_wheels"]).to_numpy()

    ov_slip_angle_front = np.arctan((ov_v_y + lf * ov_r) / (ov_v_x + eps)) - ov_delta
    ov_slip_angle_rear  = np.arctan((ov_v_y - lr * ov_r) / (ov_v_x + eps))

    ov_v_front = ov_v_x * np.cos(ov_delta) + (ov_v_y + ov_r * lf) * np.sin(ov_delta)
    ov_v_rear  = ov_v_x
    ov_slip_ratio_front = (ov_omega_front - ov_v_front) / (np.maximum(ov_omega_front, ov_v_front) + eps)
    ov_slip_ratio_rear  = (ov_omega_rear  - ov_v_rear)  / (np.maximum(ov_omega_rear,  ov_v_rear)  + eps)

# ── Evaluate distance model on policy slips ───────────────────────────────────
ov_distances = None
if PLOT_DISTANCE_EVAL and PLOT_OVERLAY:
    import torch
    from ldm.systems.commons.distance_model import DistanceModel
    _dist_model = DistanceModel.from_checkpoint(DISTANCE_MODEL_PATH)
    _ov_slips = np.column_stack([ov_slip_angle_front, ov_slip_ratio_front,
                                 ov_slip_angle_rear,  ov_slip_ratio_rear])
    with torch.no_grad():
        ov_distances = _dist_model(
            torch.tensor(_ov_slips, dtype=torch.float32)
        ).squeeze(-1).numpy()
    frac_ood = (ov_distances > DISTANCE_THRESHOLD).mean() * 100
    print(f"\nPolicy distance to dataset ({OVERLAY_LABEL}):")
    print(f"  min={ov_distances.min():.4f}  max={ov_distances.max():.4f}  "
          f"mean={ov_distances.mean():.4f}  std={ov_distances.std():.4f}")
    print(f"  fraction above threshold {DISTANCE_THRESHOLD}: {frac_ood:.1f}%")

if PLOT_STATE_VARIABLES:
    fig_state, axs_state = plt.subplots(3, 1, sharex=True)
    fig_state.suptitle("Vehicle state variables over time")
    for i, rid in enumerate(unique_runs):
        mask = run_id == rid
        idx  = np.where(mask)[0]
        #color = run_colors[rid]
        color = cmap(i % 10)
        axs_state[0].plot(idx, df["v_x"].to_numpy()[mask], color=color, label=f"run {rid}")
        axs_state[1].plot(idx, df["v_y"].to_numpy()[mask], color=color, label=f"run {rid}")
        axs_state[2].plot(idx, df["r"].to_numpy()[mask],   color=color, label=f"run {rid}")

    axs_state[0].set_ylabel("v_x")
    axs_state[1].set_ylabel("v_y")
    axs_state[2].set_ylabel("r")
    axs_state[2].set_xlabel("Time step")

    if PLOT_OVERLAY:
        idx_ov = np.arange(len(df_ov))
        axs_state[0].plot(idx_ov, ov_v_x, color=OVERLAY_COLOR, lw=1.2,
                          linestyle="--", label=OVERLAY_LABEL)
        axs_state[1].plot(idx_ov, ov_v_y, color=OVERLAY_COLOR, lw=1.2,
                          linestyle="--", label=OVERLAY_LABEL)
        axs_state[2].plot(idx_ov, ov_r,   color=OVERLAY_COLOR, lw=1.2,
                          linestyle="--", label=OVERLAY_LABEL)
        axs_state[0].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show()

## ── Spectrograms of v_x, v_y, r ──────────────────────────────────────────────
fs = 100.

from scipy.signal import welch

NFFT     = 256
NOVERLAP = 192

_spec_signals = [
    (v_x, "v_x  [m/s]"),
    (v_y, "v_y  [m/s]"),
    (r,   "r  [rad/s]"),
]

# ── Per-run Welch PSD averaged across all runs ────────────────────────────────
if PLOT_PSD:
    fig_spec, axs_spec = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig_spec.suptitle(
        f"Mean PSD across runs  (fs = {fs:.0f} Hz, Welch NFFT={NFFT})", fontsize=13
    )

    for ax, (sig, ylabel) in zip(axs_spec, _spec_signals):
        psds = []
        for rid in unique_runs:
            mask = run_id == rid
            seg  = sig[mask]
            seg  = seg - seg.mean()          # remove DC
            if len(seg) < NFFT:
                continue
            freqs, psd = welch(seg, fs=fs, nperseg=NFFT, noverlap=NOVERLAP,
                               window="hann", scaling="density")
            psds.append(psd)

        psds    = np.array(psds)             # (n_runs, n_freqs)
        mean_psd = psds.mean(axis=0)
        std_psd  = psds.std(axis=0)

        ax.semilogy(freqs, mean_psd, color="steelblue", lw=1.8, label="mean PSD")
        ax.fill_between(freqs,
                        np.maximum(mean_psd - std_psd, 1e-12),
                        mean_psd + std_psd,
                        alpha=0.3, color="steelblue", label="±1 std")
        ax.set_ylabel(f"PSD  [{ylabel.split('[')[1].rstrip(']')}²/Hz]", fontsize=9)
        ax.set_title(ylabel.split('[')[0].strip())
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    axs_spec[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

# ── Slip angles ───────────────────────────────────────────────────────────────
#   alpha_f = atan((v_y + lf*r) / (v_x + eps)) - delta
#   alpha_r = atan((v_y - lr*r) / (v_x + eps))
slip_angle_front = np.arctan((v_y + lf * r) / (v_x + eps)) - delta
slip_angle_rear  = np.arctan((v_y - lr * r) / (v_x + eps))

# ── Slip ratios ───────────────────────────────────────────────────────────────
#   kappa_f = (omega_f - v_front) / (max(omega_f, v_front) + eps)
#   kappa_r = (omega_r - v_x)     / (max(omega_r, v_x)     + eps)
v_front = v_x * np.cos(delta) + (v_y + r * lf) * np.sin(delta)
v_rear  = v_x

slip_ratio_front = (omega_front - v_front) / (np.maximum(omega_front, v_front) + eps)
slip_ratio_rear  = (omega_rear  - v_rear)  / (np.maximum(omega_rear,  v_rear)  + eps)

# ── Print summary stats ───────────────────────────────────────────────────────
def summary(name, arr):
    print(f"{name}:  min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")

print(f"Dataset: {dataset_path}  ({len(df)} samples)\n")
summary("slip_angle_front [rad]", slip_angle_front)
summary("slip_angle_rear  [rad]", slip_angle_rear)
summary("slip_ratio_front      ", slip_ratio_front)
summary("slip_ratio_rear       ", slip_ratio_rear)

if PLOT_OVERLAY:
    print(f"\nOverlay: {OVERLAY_PATH}  ({len(df_ov)} samples)")
    summary("[ov] slip_angle_front [rad]", ov_slip_angle_front)
    summary("[ov] slip_angle_rear  [rad]", ov_slip_angle_rear)
    summary("[ov] slip_ratio_front      ", ov_slip_ratio_front)
    summary("[ov] slip_ratio_rear       ", ov_slip_ratio_rear)

# ── Plot histograms ───────────────────────────────────────────────────────────
if PLOT_SLIP_HISTOGRAMS:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Slip angle & slip ratio histograms – VW Golf test set", fontsize=14)

    bins = 100

    axes[0, 0].hist(np.degrees(slip_angle_front), bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0, 0].set_title("Front slip angle")
    axes[0, 0].set_xlabel("Slip angle [deg]")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(np.degrees(slip_angle_rear), bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0, 1].set_title("Rear slip angle")
    axes[0, 1].set_xlabel("Slip angle [deg]")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(slip_ratio_front, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1, 0].set_title("Front slip ratio")
    axes[1, 0].set_xlabel("Slip ratio [-]")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(slip_ratio_rear, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1, 1].set_title("Rear slip ratio")
    axes[1, 1].set_xlabel("Slip ratio [-]")
    axes[1, 1].set_ylabel("Count")

    if PLOT_OVERLAY:
        hist_kw = dict(bins=bins, color=OVERLAY_COLOR, edgecolor="white",
                       linewidth=0.3, alpha=0.55, label=OVERLAY_LABEL)
        axes[0, 0].hist(np.degrees(ov_slip_angle_front), **hist_kw)
        axes[0, 1].hist(np.degrees(ov_slip_angle_rear),  **hist_kw)
        axes[1, 0].hist(ov_slip_ratio_front,             **hist_kw)
        axes[1, 1].hist(ov_slip_ratio_rear,              **hist_kw)
        for ax in axes.flat:
            ax.legend(fontsize=8)

    plt.tight_layout()
    #plt.savefig("experiments/imgs/slip_histograms.png", dpi=150)
    plt.show()

# ── 2-D scatter plot configuration ──────────────────────────────────────────
# Set FIT_MODE to "gaussian" for a single 2-D Gaussian,
# or "gmm" for a Gaussian Mixture Model.
FIT_MODE         = "gmm"          # "gaussian" | "gmm"
GMM_N_COMPONENTS = 7              # number of GMM components (only used when FIT_MODE="gmm")
MODEL_COLOR      = "#2CA02C"      

# ── 2-D scatter plots (SR vs SA) with fitted density ─────────────────────────
def _grid_from_data(data, n=200, margin=3.5):
    """Return a meshgrid spanning ±margin*σ around the data mean."""
    mu  = data.mean(axis=0)
    std = data.std(axis=0)
    g0  = np.linspace(mu[0] - margin * std[0], mu[0] + margin * std[0], n)
    g1  = np.linspace(mu[1] - margin * std[1], mu[1] + margin * std[1], n)
    G0, G1 = np.meshgrid(g0, g1)
    return G0, G1


def _contour_single_gaussian(ax, data, color, model_color=MODEL_COLOR):
    """Fit a single 2-D Gaussian and draw 1/2/3-σ contours."""
    mu  = data.mean(axis=0)
    cov = np.cov(data.T)
    sig = np.sqrt(np.diag(cov))

    G0, G1 = _grid_from_data(data)
    pos    = np.dstack([G0, G1])
    rv     = multivariate_normal(mean=mu, cov=cov)
    pdf    = rv.pdf(pos)

    # 1-σ/2-σ/3-σ level sets: p_peak * exp(-k²/2)
    levels = sorted(rv.pdf(mu) * np.exp(-0.5 * k**2) for k in [1, 2, 3])
    cs = ax.contour(G0, G1, pdf, levels=levels,
                    colors=[model_color] * 3,
                    linewidths=[2.5, 2.0, 1.5],
                    linestyles=["solid", "dashed", "dotted"],
                    alpha=0.9)
    fmt = {lv: lbl for lv, lbl in zip(sorted(levels), ["3σ", "2σ", "1σ"])}
    ax.clabel(cs, cs.levels, fmt=fmt, fontsize=9)

    info = (f"μ = ({mu[0]:.4f}, {mu[1]:.2f}°)\n"
            f"σ = ({sig[0]:.4f}, {sig[1]:.2f}°)\n"
            f"ρ = {cov[0,1]/(sig[0]*sig[1]):.3f}")
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def _contour_gmm(ax, data, color, n_components, model_color=MODEL_COLOR):
    """Fit a GMM and draw density contours at the 50/90/99 % quantile levels."""
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=0, max_iter=500)
    gmm.fit(data)

    G0, G1 = _grid_from_data(data)
    pos    = np.dstack([G0, G1])          # (200, 200, 2)
    flat   = pos.reshape(-1, 2)
    log_p  = gmm.score_samples(flat)      # log-density at each grid point
    pdf    = np.exp(log_p).reshape(G0.shape)

    # Choose contour levels as quantiles of the density over the grid
    # so they are data-adaptive regardless of the number of components
    q_levels = np.quantile(pdf[pdf > 0], [0.50, 0.80, 0.95])
    q_labels  = ["50%", "80%", "95%"]
    levels   = sorted(set(q_levels))      # ascending, unique

    cs = ax.contour(G0, G1, pdf, levels=levels,
                    colors=[model_color] * len(levels),
                    linewidths=[1.5, 2.0, 2.5],
                    linestyles=["dotted", "dashed", "solid"],
                    alpha=0.9)
    fmt = {lv: lbl for lv, lbl in zip(sorted(levels), q_labels[:len(levels)])}
    ax.clabel(cs, cs.levels, fmt=fmt, fontsize=9)

    # Print per-component summary
    lines = [f"GMM  K={n_components}  (BIC={gmm.bic(data):.0f})"]
    for k in range(n_components):
        w  = gmm.weights_[k]
        m  = gmm.means_[k]
        cv = gmm.covariances_[k]
        s  = np.sqrt(np.diag(cv))
        lines.append(f"  #{k+1} w={w:.2f}  μ=({m[0]:.4f},{m[1]:.2f}°)  "
                     f"σ=({s[0]:.4f},{s[1]:.2f}°)")
    info = "\n".join(lines)
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def plot_sr_sa_scatter(ax, sr, sa_deg, color, title,
                       fit_mode=FIT_MODE, n_components=GMM_N_COMPONENTS,
                       model_color=MODEL_COLOR):
    """Scatter SR/SA pairs and overlay a fitted density (Gaussian or GMM)."""
    ax.scatter(sr, sa_deg, s=2, alpha=0.3, color=color, rasterized=True, label="data")

    data = np.column_stack([sr, sa_deg])

    if fit_mode == "gaussian":
        _contour_single_gaussian(ax, data, color, model_color=model_color)
        mode_label = "single Gaussian"
    elif fit_mode == "gmm":
        _contour_gmm(ax, data, color, n_components, model_color=model_color)
        mode_label = f"GMM  (K={n_components})"
    else:
        raise ValueError(f"Unknown fit_mode '{fit_mode}'. Use 'gaussian' or 'gmm'.")

    ax.set_title(f"{title}  [{mode_label}]")
    ax.set_xlabel("Slip ratio [-]")
    ax.set_ylabel("Slip angle [deg]")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)


if PLOT_SR_SA_SCATTER:
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(f"SR / SA joint distribution  –  fit: {FIT_MODE}", fontsize=14)

    plot_sr_sa_scatter(axes2[0], slip_ratio_front, np.degrees(slip_angle_front),
                       color="steelblue", title="Front wheel")
    plot_sr_sa_scatter(axes2[1], slip_ratio_rear,  np.degrees(slip_angle_rear),
                       color="steelblue",    title="Rear wheel")

    if PLOT_OVERLAY:
        ov_t = np.arange(len(df_ov))          # time index used for colour coding
        scatter_kw = dict(s=8, alpha=0.7, marker="^", rasterized=True,
                          cmap="plasma", c=ov_t, zorder=3)
        sc0 = axes2[0].scatter(ov_slip_ratio_front, np.degrees(ov_slip_angle_front), **scatter_kw)
        sc1 = axes2[1].scatter(ov_slip_ratio_rear,  np.degrees(ov_slip_angle_rear),  **scatter_kw)
        fig2.colorbar(sc0, ax=axes2[0], label="Time index", pad=0.02)
        fig2.colorbar(sc1, ax=axes2[1], label="Time index", pad=0.02)
        # Proxy artist so the overlay still appears in the legend
        from matplotlib.lines import Line2D
        proxy = Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                       markersize=6, label=OVERLAY_LABEL)
        axes2[0].legend(handles=[*axes2[0].get_legend_handles_labels()[0], proxy],
                        loc="upper right", markerscale=3, fontsize=8)
        axes2[1].legend(handles=[*axes2[1].get_legend_handles_labels()[0], proxy],
                        loc="upper right", markerscale=3, fontsize=8)

    plt.tight_layout()
    #plt.savefig("experiments/imgs/slip_sr_sa_scatter.png", dpi=150)
    plt.show()

# ── Distance-to-dataset visualisation ────────────────────────────────────────
if PLOT_DISTANCE_EVAL and PLOT_OVERLAY and ov_distances is not None:
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    # Clamp colour range at the 99th percentile so a few extreme OOD points
    # don't wash out the colour scale across the rest of the trajectory.
    d_lo  = ov_distances.min()
    d_hi  = np.percentile(ov_distances, 99)
    norm_d = Normalize(vmin=d_lo, vmax=d_hi)
    cmap_d = plt.get_cmap("RdYlGn_r")   # green = close, red = far

    ood_mask = ov_distances > DISTANCE_THRESHOLD

    fig_de, axes_de = plt.subplots(2, 2, figsize=(16, 12))
    fig_de.suptitle(
        f"Policy distance to training dataset  –  {OVERLAY_LABEL}\n"
        f"(threshold = {DISTANCE_THRESHOLD},  OOD fraction = {frac_ood:.1f}%)",
        fontsize=13,
    )

    # ── Row 0: wheel SA/SR scatter coloured by distance ──────────────────────
    wheel_cfg = [
        (ov_slip_ratio_front, ov_slip_angle_front,
         slip_ratio_front,    slip_angle_front,    "Front wheel"),
        (ov_slip_ratio_rear,  ov_slip_angle_rear,
         slip_ratio_rear,     slip_angle_rear,     "Rear wheel"),
    ]
    for col, (sr_ov, sa_ov, sr_tr, sa_tr, wtitle) in enumerate(wheel_cfg):
        ax = axes_de[0, col]

        # Training distribution as a faint background
        ax.scatter(sr_tr, np.degrees(sa_tr), s=1, alpha=0.07, c="steelblue",
                   rasterized=True, label="training data")

        # Policy trajectory coloured by distance
        sc = ax.scatter(sr_ov, np.degrees(sa_ov), s=8, alpha=0.75,
                        c=ov_distances, cmap=cmap_d, norm=norm_d,
                        rasterized=True, zorder=3,
                        label="policy (coloured by distance)")

        # Highlight OOD points with a cross marker
        if ood_mask.any():
            ax.scatter(sr_ov[ood_mask], np.degrees(sa_ov[ood_mask]),
                       s=30, alpha=0.9, c="red", marker="x", linewidths=0.8,
                       rasterized=True, zorder=4,
                       label=f"OOD  (d > {DISTANCE_THRESHOLD})")

        plt.colorbar(sc, ax=ax, label="distance to dataset")
        ax.set_xlabel("Slip ratio [−]")
        ax.set_ylabel("Slip angle [deg]")
        ax.set_title(wtitle)
        ax.legend(loc="upper right", fontsize=8, markerscale=3)

    # ── Row 1 left: distance time series ─────────────────────────────────────
    ax_ts = axes_de[1, 0]
    t = np.arange(len(ov_distances))
    # Colour each segment by its distance value
    for k in range(len(t) - 1):
        c = cmap_d(norm_d(ov_distances[k]))
        ax_ts.plot(t[k:k+2], ov_distances[k:k+2], color=c, lw=0.9)
    ax_ts.axhline(DISTANCE_THRESHOLD, color="red", ls="--", lw=1.5,
                  label=f"threshold = {DISTANCE_THRESHOLD}")
    ax_ts.fill_between(t, ov_distances, DISTANCE_THRESHOLD,
                       where=ood_mask, color="red", alpha=0.20,
                       label="OOD region")
    ax_ts.set_xlabel("Time step")
    ax_ts.set_ylabel("Predicted distance")
    ax_ts.set_title("Distance to training set over time")
    sm = ScalarMappable(cmap=cmap_d, norm=norm_d)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_ts, label="distance")
    ax_ts.legend(fontsize=8)

    # ── Row 1 right: distance histogram ──────────────────────────────────────
    ax_hist = axes_de[1, 1]
    n_bins = min(80, max(20, len(ov_distances) // 50))
    ax_hist.hist(ov_distances, bins=n_bins, color="steelblue",
                 edgecolor="white", linewidth=0.3, density=True,
                 label="policy distances")
    ax_hist.axvline(DISTANCE_THRESHOLD, color="red", ls="--", lw=2,
                    label=f"threshold = {DISTANCE_THRESHOLD}")
    ax_hist.axvspan(DISTANCE_THRESHOLD, ov_distances.max(),
                    alpha=0.12, color="red", label=f"OOD ({frac_ood:.1f}%)")
    ax_hist.set_xlabel("Predicted distance to dataset")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Distance distribution  ({frac_ood:.1f}% above threshold)")
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # ── 2-D slices of the 4-D slip space coloured by distance ────────────────
    # Policy slip vector: [sa_front, sr_front, sa_rear, sr_rear]
    ov_slips_4d = np.column_stack([ov_slip_angle_front, ov_slip_ratio_front,
                                   ov_slip_angle_rear,  ov_slip_ratio_rear])
    tr_slips_4d = np.column_stack([slip_angle_front, slip_ratio_front,
                                   slip_angle_rear,  slip_ratio_rear])

    SLICE_DIM_NAMES = [
        "SA front [rad]", "SR front [−]",
        "SA rear [rad]",  "SR rear [−]",
    ]

    from itertools import combinations as _comb
    pairs = list(_comb(range(4), 2))   # 6 pairs

    fig_sl, axes_sl = plt.subplots(2, 3, figsize=(20, 12))
    fig_sl.suptitle(
        f"Policy OOD analysis – 2-D slip-space slices  ({OVERLAY_LABEL})\n"
        f"threshold = {DISTANCE_THRESHOLD},  OOD fraction = {frac_ood:.1f}%",
        fontsize=13,
    )
    axes_sl_flat = axes_sl.ravel()

    for idx, (i, j) in enumerate(pairs):
        ax = axes_sl_flat[idx]

        # Training data as a faint background cloud
        ax.scatter(tr_slips_4d[:, i], tr_slips_4d[:, j],
                   s=1, alpha=0.06, c="steelblue", rasterized=True,
                   label="training data")

        # In-distribution policy points coloured by distance
        in_mask = ~ood_mask
        if in_mask.any():
            sc = ax.scatter(
                ov_slips_4d[in_mask, i], ov_slips_4d[in_mask, j],
                s=8, alpha=0.75,
                c=ov_distances[in_mask], cmap=cmap_d, norm=norm_d,
                rasterized=True, zorder=3,
                label="policy (in-dist)",
            )

        # OOD policy points – prominent red crosses
        if ood_mask.any():
            ax.scatter(
                ov_slips_4d[ood_mask, i], ov_slips_4d[ood_mask, j],
                s=40, alpha=0.9, c=ov_distances[ood_mask],
                cmap=cmap_d, norm=norm_d,
                marker="x", linewidths=1.2,
                rasterized=True, zorder=4,
                label=f"OOD  (d > {DISTANCE_THRESHOLD})",
            )
            # Black outline ring so OOD points stand out even on colourful bg
            ax.scatter(
                ov_slips_4d[ood_mask, i], ov_slips_4d[ood_mask, j],
                s=80, alpha=0.4, facecolors="none", edgecolors="red",
                linewidths=0.8, rasterized=True, zorder=3,
            )

        ax.set_xlabel(SLICE_DIM_NAMES[i])
        ax.set_ylabel(SLICE_DIM_NAMES[j])
        ax.set_title(f"{SLICE_DIM_NAMES[i]}  vs  {SLICE_DIM_NAMES[j]}")
        ax.legend(loc="upper right", fontsize=7, markerscale=3)

    # Shared colour-bar on the right
    sm2 = ScalarMappable(cmap=cmap_d, norm=norm_d)
    sm2.set_array([])
    fig_sl.colorbar(sm2, ax=axes_sl_flat, label="distance to dataset",
                    fraction=0.015, pad=0.02)

    plt.tight_layout()
    plt.show()
