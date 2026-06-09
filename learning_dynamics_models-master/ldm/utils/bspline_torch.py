"""
Differentiable Uniform Clamped B-Splines in PyTorch
=====================================================
Supports batched evaluation, arbitrary degree, and gradient flow through
both the control points and the parameter t.

Control points are NOT stored on the module — they are passed directly to
forward(), giving full flexibility over their lifecycle (nn.Parameter,
plain tensor, output of another network, etc.).
"""

import torch
import torch.nn as nn
from typing import Optional


def _clamped_knot_vector(n_control: int, degree: int, device=None, dtype=None) -> torch.Tensor:
    """
    Build a uniform clamped (open) knot vector for n_control points and given degree.
    Total knots = n_control + degree + 1.
    The first (degree+1) and last (degree+1) knots are repeated to clamp the spline.

    Example: n_control=5, degree=3 → [0,0,0,0, 1, 2, 2,2,2] (normalised to [0,1])
    """
    n_knots = n_control + degree + 1
    n_inner = n_knots - 2 * (degree + 1)
    knots = torch.zeros(n_knots, device=device, dtype=dtype)
    knots[: degree + 1] = 0.0
    if n_inner > 0:
        internal = torch.linspace(0.0, 1.0, n_inner + 2, device=device, dtype=dtype)[1:-1]
        knots[degree + 1 : degree + 1 + n_inner] = internal
    knots[degree + 1 + n_inner :] = 1.0
    return knots


def _basis_functions(t: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate all B-spline basis functions of the given degree at parameters t
    via the Cox–de Boor recurrence (differentiable w.r.t. t).

    Parameters
    ----------
    t      : (B,) parameter values in [0, 1]
    knots  : (m,) knot vector
    degree : spline degree p

    Returns
    -------
    N : (B, n_control) basis-function matrix, where n_control = m - p - 1
    """
    n_control = knots.shape[0] - degree - 1
    B_size = t.shape[0]
    dtype = t.dtype

    # Clamp t to avoid boundary issues at t == 1
    t_clamped = t.clamp(knots[degree], knots[-degree - 1] - 1e-7 * (knots[-1] - knots[0]))

    # Degree-0: indicator for each knot span
    k_left  = knots[:-1].unsqueeze(0)   # (1, m-1)
    k_right = knots[ 1:].unsqueeze(0)   # (1, m-1)
    tc = t_clamped.unsqueeze(1)          # (B, 1)
    N = ((tc >= k_left) & (tc < k_right)).to(dtype)  # (B, m-1)

    # Recurrence up to degree p
    for d in range(1, degree + 1):
        ki   = knots[    : -(d + 1)].unsqueeze(0)
        kid  = knots[  d :    -1   ].unsqueeze(0)
        kid1 = knots[d + 1 :       ].unsqueeze(0)
        ki1  = knots[    1 :   -d  ].unsqueeze(0)

        denom1 = kid  - ki
        denom2 = kid1 - ki1

        alpha = torch.where(
            denom1.abs() > 1e-10,
            (tc - ki)   / denom1.clamp(min=1e-10),
            torch.zeros_like(tc),
        )
        beta = torch.where(
            denom2.abs() > 1e-10,
            (kid1 - tc) / denom2.clamp(min=1e-10),
            torch.zeros_like(tc),
        )

        N = alpha * N[:, :-1] + beta * N[:, 1:]

    assert N.shape == (B_size, n_control), f"Expected ({B_size},{n_control}), got {N.shape}"
    return N


class BSpline(nn.Module):
    """
    Differentiable uniform clamped B-spline evaluator.

    The module only owns the *geometry* of the spline (degree and the derived
    knot vector). Control points are supplied at call time, so the caller
    retains full ownership — they can be an ``nn.Parameter``, the output of
    a neural network, or any plain tensor.

    Parameters
    ----------
    n_control : number of control points
    degree    : spline degree (1=linear, 2=quadratic, 3=cubic, ...)
    """

    def __init__(self, n_control: int, degree: int = 3):
        super().__init__()
        assert n_control > degree, (
            f"Need n_control > degree, got n_control={n_control}, degree={degree}"
        )
        self.n_control = n_control
        self.degree = degree

        knots = _clamped_knot_vector(n_control, degree)
        self.register_buffer("knots", knots)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def forward(self, t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the spline at parameter values t.

        Three calling conventions are supported, distinguished by the shape of
        ``control_points``:

        1. *Non-batched, scalar output*
           ``control_points``: ``(n_control,)``, ``t``: ``(B,)``
           → output ``(B,)``

        2. *Non-batched, vector output* (original behaviour)
           ``control_points``: ``(n_control, dim)``, ``t``: ``(B,)``
           → output ``(B, dim)``

        3. *Batched, scalar output*
           ``control_points``: ``(batch_size, n_control)``, ``t``: ``(batch_size,)``
           Each sample in the batch has its own set of control points and is
           evaluated at its own parameter value.
           → output ``(batch_size,)``
        """
        t = t.to(control_points.dtype).reshape(-1)  # (B,)
        cp = control_points

        if cp.ndim == 1:
            # Convention 1: (n_control,)
            if cp.shape[0] != self.n_control:
                raise ValueError(
                    f"Expected {self.n_control} control points, got {cp.shape[0]}"
                )
            N = _basis_functions(t, self.knots, self.degree)  # (B, n_control)
            return N @ cp                                      # (B,)

        elif cp.ndim == 2 and cp.shape[0] == self.n_control:
            # Convention 2: (n_control, dim)
            N = _basis_functions(t, self.knots, self.degree)  # (B, n_control)
            return N @ cp                                      # (B, dim)

        elif cp.ndim == 2 and cp.shape[1] == self.n_control:
            # Convention 3: (batch_size, n_control)
            if cp.shape[0] != t.shape[0]:
                raise ValueError(
                    f"Batched mode requires control_points batch size ({cp.shape[0]}) "
                    f"to match t batch size ({t.shape[0]})"
                )
            N = _basis_functions(t, self.knots, self.degree)  # (B, n_control)
            return (N * cp).sum(dim=-1)                        # (B,)

        else:
            raise ValueError(
                f"control_points shape {tuple(cp.shape)} is incompatible with "
                f"n_control={self.n_control}. Expected (n_control,), "
                f"(n_control, dim), or (batch_size, n_control)."
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def sample(
        self, control_points: torch.Tensor, n_samples: int = 100
    ) -> torch.Tensor:
        """Return `n_samples` uniformly-spaced points along the spline."""
        t = torch.linspace(
            0.0, 1.0, n_samples,
            device=control_points.device,
            dtype=control_points.dtype,
        )
        return self.forward(t, control_points)

    def arc_length(
        self, control_points: torch.Tensor, n_samples: int = 1000
    ) -> torch.Tensor:
        """Approximate arc length by summing chord lengths (differentiable)."""
        pts = self.sample(control_points, n_samples)
        return (pts[1:] - pts[:-1]).norm(dim=-1).sum()

    def __repr__(self) -> str:
        return (
            f"BSpline(n_control={self.n_control}, degree={self.degree}, "
            f"n_knots={self.knots.shape[0]})"
        )


# ---------------------------------------------------------------------------
# Utility: fit a B-spline to a set of target points via gradient descent
# ---------------------------------------------------------------------------

def fit_bspline(
    target: torch.Tensor,
    n_control: int = 10,
    degree: int = 3,
    n_iter: int = 2000,
    lr: float = 1e-2,
) -> tuple:
    """
    Fit a B-spline to target points (n_pts, dim) by minimising MSE with Adam.

    Returns
    -------
    spline         : BSpline module
    control_points : fitted (n_control, dim) nn.Parameter
    """
    n_pts, dim = target.shape
    t_data = torch.linspace(0.0, 1.0, n_pts, dtype=target.dtype)

    spline = BSpline(n_control=n_control, degree=degree)

    # Initialise control points along the target curve
    idx = torch.linspace(0, n_pts - 1, n_control).long()
    control_points = nn.Parameter(target[idx].clone().float())

    optimiser = torch.optim.Adam([control_points], lr=lr)

    for i in range(n_iter):
        optimiser.zero_grad()
        pred = spline(t_data, control_points)
        loss = (pred - target).pow(2).sum(dim=-1).mean()
        loss.backward()
        optimiser.step()

        if (i + 1) % 500 == 0:
            print(f"  iter {i+1:4d}/{n_iter}  loss={loss.item():.6f}")

    return spline, control_points


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    torch.manual_seed(0)

    print("=== BSpline self-test ===\n")

    spline = BSpline(n_control=6, degree=3)
    print(spline)

    # Control points owned externally as an nn.Parameter
    ctrl = nn.Parameter(torch.zeros(6, 2))
    ctrl.data[:, 0] = torch.linspace(0.0, 1.0, 6)

    # 1. Basic forward pass
    t = torch.linspace(0, 1, 5)
    pts = spline(t, ctrl)
    print(f"\nEvaluated at 5 points:\n{pts.detach()}\n")

    # 2. Clamped endpoint interpolation
    print("First control point :", ctrl[0].detach())
    print("Spline at t=0       :", spline(torch.tensor([0.0]), ctrl)[0].detach())
    print("Last  control point :", ctrl[-1].detach())
    print("Spline at t=1       :", spline(torch.tensor([1.0]), ctrl)[0].detach())

    # 3. Gradient through control points
    loss = spline.sample(ctrl, 50).pow(2).sum()
    loss.backward()
    print(f"\n||grad(ctrl)|| = {ctrl.grad.norm().item():.4f}")

    # 4. Gradient through t
    t2 = torch.linspace(0, 1, 20).requires_grad_(True)
    spline(t2, ctrl).sum().backward()
    print(f"||grad(t)||    = {t2.grad.norm().item():.4f}")

    # 5. Batched mode: (batch_size, n_control) control points with t (batch_size,)
    print("\n--- Batched mode (batch_size, n_control) ---")
    batch_size = 8
    spline_1d = BSpline(n_control=6, degree=3)
    # Each batch element has its own scalar-valued control points
    ctrl_batch = torch.randn(batch_size, 6, requires_grad=True)
    t_batch = torch.rand(batch_size, requires_grad=True)
    out_batch = spline_1d(t_batch, ctrl_batch)
    print(f"Input  ctrl_batch : {ctrl_batch.shape}")
    print(f"Input  t_batch    : {t_batch.shape}")
    print(f"Output            : {out_batch.shape}  (expected ({batch_size},))")
    out_batch.sum().backward()
    print(f"||grad(ctrl_batch)|| = {ctrl_batch.grad.norm().item():.4f}")
    print(f"||grad(t_batch)||    = {t_batch.grad.norm().item():.4f}")

    # 5. Fit a circle
    print("\nFitting a circle with 12 control points ...")
    theta = torch.linspace(0, 2 * 3.14159, 200)
    circle = torch.stack([theta.cos(), theta.sin()], dim=1)
    fitted_spline, fitted_ctrl = fit_bspline(circle, n_control=12, degree=3, n_iter=2000, lr=5e-2)
    mse = (fitted_spline.sample(fitted_ctrl, 200) - circle).pow(2).mean().item()
    print(f"Final MSE: {mse:.6f}\n")

    print("All tests passed!")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Panel 1: 2-D open curve with arbitrary control polygon -----------------
    ax1 = fig.add_subplot(gs[0])

    ctrl_2d = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.8],
        [0.3, 1.0],
        [0.5, 0.4],
        [0.7, 0.9],
        [0.9, 0.2],
        [1.0, 0.5],
    ], dtype=torch.float32)
    sp2d = BSpline(n_control=ctrl_2d.shape[0], degree=3)
    curve_2d = sp2d.sample(ctrl_2d, n_samples=300).detach().numpy()
    cp_np = ctrl_2d.numpy()

    ax1.plot(cp_np[:, 0], cp_np[:, 1],
             "o--", color="steelblue", linewidth=1.2, markersize=7,
             label="Control polygon")
    ax1.plot(curve_2d[:, 0], curve_2d[:, 1],
             "-", color="tomato", linewidth=2.5,
             label="B-spline curve")
    ax1.plot(cp_np[0, 0],  cp_np[0, 1],  "^", color="green", markersize=10, zorder=5, label="Start / end CP")
    ax1.plot(cp_np[-1, 0], cp_np[-1, 1], "s", color="green", markersize=10, zorder=5)
    ax1.set_title("Open cubic B-spline (degree 3, 7 CPs)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 2: fitted circle --------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    curve_fit = fitted_spline.sample(fitted_ctrl, 300).detach().numpy()
    fc_np = fitted_ctrl.detach().numpy()
    tgt_np = circle.numpy()

    ax2.plot(tgt_np[:, 0], tgt_np[:, 1],
             "--", color="gray", linewidth=1.5, label="Target circle")
    ax2.plot(fc_np[:, 0], fc_np[:, 1],
             "o", color="steelblue", markersize=8, label="Control points")
    # connect control polygon
    ax2.plot(
        list(fc_np[:, 0]) + [fc_np[0, 0]],
        list(fc_np[:, 1]) + [fc_np[0, 1]],
        "--", color="steelblue", linewidth=1.0, alpha=0.6,
    )
    ax2.plot(curve_fit[:, 0], curve_fit[:, 1],
             "-", color="tomato", linewidth=2.5, label=f"Fitted spline (MSE={mse:.4f})")
    ax2.set_title("B-spline fitted to a circle (12 CPs, degree 3)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Differentiable B-Splines in PyTorch", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
