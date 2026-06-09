import torch

def get_integrator(integration_method):
    if integration_method not in ["rk4", "euler"]:
        raise ValueError(f"Invalid integration method: {integration_method}. Choose 'rk4' or 'euler'.")
    else:
        return eval(integration_method)
    
def rk4(f, xu, dt):
    t = torch.zeros(1, dtype=torch.float32, device=xu.device)
    k1 = f(t, xu)
    k2 = f(t, xu + dt / 2 * k1)
    k3 = f(t, xu + dt / 2 * k2)
    k4 = f(t, xu + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) + xu

def euler(f, xu, dt):
    t = torch.zeros(1, dtype=torch.float32, device=xu.device)
    return xu + dt * f(t, xu)
