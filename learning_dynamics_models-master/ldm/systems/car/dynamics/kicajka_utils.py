import torch

from ldm.utils.bspline_torch import _basis_functions


def compute_force(bspline, mod, scale, norm, res, cps):
    # Avoid the unnecessary repeat() when cps is 1-D: BSpline Convention 1
    # handles (n_control,) directly via N @ cps (no allocation needed).
    t = torch.clip(mod * scale, 0., 1.0)
    F = bspline(t, cps)
    return (norm / res) * F


def compute_forces_quad(
    bspline,
    t_xf, cps_xf, norm_xf,
    t_yf, cps_yf, norm_yf,
    t_xr, cps_xr, norm_xr,
    t_yr, cps_yr, norm_yr,
    S_f, S_r,
):
    """Evaluate all four tire forces in a *single* _basis_functions call.

    By concatenating the four parameter vectors (front-Fx, front-Fy,
    rear-Fx, rear-Fy) the Cox-de Boor recurrence is executed only once
    instead of four times, giving ~2-3× speedup on the BSpline part.

    Parameters
    ----------
    bspline          : BSpline module (owns the knot buffer)
    t_xf, t_yf,
    t_xr, t_yr       : (B,) pre-clipped t values in [0, 1]
    cps_xf, cps_yf,
    cps_xr, cps_yr   : (n_control,) or (B, n_control) control-point tensors
    norm_xf, norm_yf,
    norm_xr, norm_yr : (B,) normalised-slip scalars
    S_f, S_r         : (B,) resultant-slip magnitudes (+ eps)

    Returns
    -------
    fric_xf, fric_yf, fric_xr, fric_yr : (B,) friction coefficients
    """
    B = t_xf.shape[0]
    t_all = torch.cat([t_xf, t_yf, t_xr, t_yr])          # (4B,)
    N = _basis_functions(t_all, bspline.knots, bspline.degree)  # (4B, n_ctrl)

    def _eval(N_slice, cps):
        if cps.ndim == 1:
            return N_slice @ cps               # (B,)
        return (N_slice * cps).sum(-1)         # (B,) – batched cps

    Fx_f = (norm_xf / S_f) * _eval(N[:B],       cps_xf)
    Fy_f = (norm_yf / S_f) * _eval(N[B:2*B],    cps_yf)
    Fx_r = (norm_xr / S_r) * _eval(N[2*B:3*B],  cps_xr)
    Fy_r = (norm_yr / S_r) * _eval(N[3*B:],     cps_yr)

    return Fx_f, -Fy_f, Fx_r, -Fy_r

