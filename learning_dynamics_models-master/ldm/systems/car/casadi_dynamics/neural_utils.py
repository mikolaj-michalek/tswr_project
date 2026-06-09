import torch
import casadi as ca

def _torch_sequential_to_casadi(seq, x_sym):
    """
    Convert a torch.nn.Sequential composed of Linear and Tanh layers
    into a CasADi symbolic expression.

    Parameters
    ----------
    seq : torch.nn.Sequential
        PyTorch sequential model (Linear → Tanh → … → Linear).
    x_sym : ca.MX
        CasADi symbolic column vector (n_in × 1).

    Returns
    -------
    ca.MX
        CasADi symbolic column vector (n_out × 1).
    """
    h = x_sym
    for layer in seq:
        if isinstance(layer, torch.nn.Linear):
            W = layer.weight.detach().numpy()   # (out, in)
            b = layer.bias.detach().numpy()      # (out,)
            h = W @ h + b[:, None]               # (out, 1)
        elif isinstance(layer, torch.nn.Tanh):
            h = ca.tanh(h)
        elif isinstance(layer, torch.nn.ReLU):
            h = ca.fmax(h, 0.0)
        elif isinstance(layer, torch.nn.ELU):
            # ELU: x if x>0, alpha*(exp(x)-1) otherwise; default alpha=1
            alpha = getattr(layer, 'alpha', 1.0)
            h = ca.if_else(h > 0, h, alpha * (ca.exp(h) - 1))
        else:
            raise NotImplementedError(
                f"Unsupported layer type: {type(layer)}"
            )
    return h

