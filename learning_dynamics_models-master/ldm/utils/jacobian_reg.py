import torch


def hutchinson_jacobian_frobenius_sq(
    model: torch.nn.Module,
    X: torch.Tensor,
    U: torch.Tensor,
    n_x: int,
    device: torch.device,
    create_graph: bool = True,
) -> torch.Tensor:
    """Hutchinson estimator of the squared Frobenius norm of the Jacobian
    d(dx)/d([X, U]) evaluated at a single batch of (X, U) points.

    Uses one forward-mode AD pass (JVP), so cost is O(1) w.r.t. both the
    number of inputs and outputs.  The estimator is unbiased:

        E_v[ ||J v||^2 ] = ||J||_F^2    for v ~ N(0, I)

    Args:
        model:        Dynamics model callable as ``model(t, X, U) -> (dx, *)``.
        X:            State tensor of shape ``(B, state_dim)``.
        U:            Control tensor of shape ``(B, control_dim)``.
        n_x:          State dimension (used to split ``xu`` back into X and U).
        device:       Torch device.
        create_graph: If ``True`` (default), the result can be differentiated
                      w.r.t. model parameters (needed when used as a loss term).

    Returns:
        Scalar tensor: mean over the batch of ``||J v||^2``.
    """
    t = torch.zeros(1, device=device)
    xu = torch.cat([X, U], dim=-1).detach().requires_grad_(True)  # (B, n_in)
    rand = torch.randn_like(xu)
    v = rand / rand.norm(dim=-1, keepdim=True)     # (B, n_in)

    def _dyn_xu(xu_):
        dx, _, _ = model(t, xu_[:, :n_x], xu_[:, n_x:])
        return dx

    _, jvp_out = torch.autograd.functional.jvp(
        _dyn_xu, (xu,), (v,), create_graph=create_graph,
    )                                                              # (B, n_out)
    jvp_threshold = 1.
    jvp_out_norm = jvp_out.norm(dim=-1)  # (B,)
    jvp_out_violations = torch.relu(jvp_out_norm - jvp_threshold) ** 2.
    return jvp_out_violations.mean()
