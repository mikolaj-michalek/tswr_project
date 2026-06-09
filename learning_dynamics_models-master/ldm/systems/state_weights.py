import torch

def get_state_weights(system: str):
    if system == "f1tenth":
        return torch.tensor([1.0, 1.0, 1.0])
    elif system == "acrobot":
        return torch.tensor([1.0, 1.0, 1.0, 1.0])
    elif system == "vw_golf":
        return torch.tensor([1.0, 1.0, 1.0])
    else:
        raise ValueError(f"Unknown system: {system}")