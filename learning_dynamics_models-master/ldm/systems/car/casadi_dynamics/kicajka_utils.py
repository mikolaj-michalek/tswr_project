import casadi as ca

def compute_force_casadi(bspline, mod, scale, norm, res, cps):
    t = mod * scale
    t = ca.fmax(t, 0.)
    t = ca.fmin(t, 1.0)

    N = bspline(t)
    F = N.T @ cps[:, None]
    #F = bspline(t)
    return (norm / res) * F