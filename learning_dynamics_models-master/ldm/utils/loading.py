import os
import re
import torch
import yaml

from ldm.systems.handler import get_dynamics_model
from ldm.systems.commons.distance_model import DistanceModel
from ldm.systems.commons.distance_aware_wrapper import DistanceAwareDynamicsWrapper
from ldm.systems.mlp.combined_residual_model import CombinedResidualDynamicsModel
from ldm.systems.mlp.residual_model_helpers import build_models_from_cfg

def load_model_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.full_load(f)
    cfg['cfg_path'] = config_path
    return cfg


def load_model_from_config(cfg):
    system = cfg.get('system')
    dynamics_model_type = cfg.get('dynamics_model_type')
    state_extender = cfg.get('state_extender_type')
    model_path = cfg.get('model_path')
    kwargs = cfg.get('model_kwargs', {})
    base_path = os.path.dirname(cfg['cfg_path']) if 'cfg_path' in cfg else None
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_path, model_path)

    dynamics_model, state_extender = get_dynamics_model(system=system,
                                                        model_type=dynamics_model_type,
                                                        state_extender_type=state_extender,
                                                        **kwargs)
    state_dict = torch.load(model_path)
    dynamics_model.load_state_dict(state_dict, strict=False)
    return dynamics_model


def load_learned_model(config_path):
    if config_path.endswith(".yml") or config_path.endswith(".yaml"):
        cfg = load_model_config(config_path)
        dynamics_model = load_model_from_config(cfg)
    elif config_path.endswith(".pt") or config_path.endswith(".pth"):
        dynamics_model, cfg = load_dynamics_model(config_path)
    else:
        raise ValueError(f"Unsupported model file format: {config_path}")
    return dynamics_model, cfg

def get_model_config_from_model_path(model_path: str):
    model_name = os.path.basename(model_path)
    begining = model_name[:model_name.find("_gc")]
    dynamics_model_type = "_".join(begining.split("_")[:-1])
    state_extender = begining.split("_")[-1]
    if state_extender == "None":
        state_extender = None
    system = model_path.split("/")[-3]
    if system.startswith("f1tenth_new"):
        system = "f1tenth"
    if system.startswith("vw_golf_"):
        system = "vw_golf"
    model_kwargs = {}
    if dynamics_model_type.startswith("kicajka"):
        n = re.search("_n[0-9]+_", model_name).group()[2:-1]
        n_up = re.search("_nup[0-9]+_", model_name).group()[4:-1]
        model_kwargs['n'] = int(n)
        model_kwargs['n_up'] = int(n_up)
    #elif dynamics_model_type.startswith("neural_exptanh"):
    else:
        nn = re.search("_nn[0-9]+_", model_name).group()[3:-1]
        model_kwargs['nn'] = int(nn)
        #if dynamics_model_type.startswith("neural_exptanh"):
        #    model_kwargs['nn'] = 3
        
    cfg = dict(
        system=system,
        dynamics_model_type=dynamics_model_type,
        state_extender_type=state_extender,
        model_path=model_path,
        model_kwargs=model_kwargs,
    )
    return cfg

def load_dynamics_model(model_path: str):
    cfg = get_model_config_from_model_path(model_path)
    dynamics_model = load_model_from_config(cfg)
    return dynamics_model, cfg


# ── Distance-to-dataset model ──────────────────────────────────────────────────

def load_distance_model(checkpoint_path: str,
                        map_location: str = "cpu") -> DistanceModel:
    """
    Load a distance-to-dataset model from a checkpoint produced by
    ``experiments/train_distance_model.py``.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint file.
    map_location : str
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    DistanceModel
    """
    return DistanceModel.from_checkpoint(checkpoint_path,
                                         map_location=map_location)


def load_model_with_distance(
    dynamics_config_path: str,
    distance_checkpoint_path: str,
    threshold: float = 1.0,
    map_location: str = "cpu",
) -> DistanceAwareDynamicsWrapper:
    """
    Load a learned dynamics model together with a distance-to-dataset network
    and return a ``DistanceAwareDynamicsWrapper`` that dampens predictions
    outside the training distribution.

    Parameters
    ----------
    dynamics_config_path : str
        Path to a YAML config file **or** a ``.pt`` model file understood by
        ``load_learned_model``.
    distance_checkpoint_path : str
        Path to the distance-model checkpoint (``.pt``).
    threshold : float
        Distance value below which predictions are unmodified.
    map_location : str
        Torch device string.

    Returns
    -------
    DistanceAwareDynamicsWrapper
    """
    dynamics_model, _ = load_learned_model(dynamics_config_path)
    distance_model = load_distance_model(distance_checkpoint_path,
                                         map_location=map_location)
    return DistanceAwareDynamicsWrapper(
        dynamics_model=dynamics_model,
        distance_model=distance_model,
        threshold=threshold,
    )


# ── Combined base + residual model ────────────────────────────────────────────

def load_combined_residual_model(
    checkpoint_path: str,
    base_model_path: str = None,
    map_location: str = "cpu",
    freeze_base: bool = True,
) -> CombinedResidualDynamicsModel:
    """
    Load a ``CombinedResidualDynamicsModel`` from a checkpoint produced by
    ``experiments/train_residual.py``.

    The checkpoint contains both model weights **and** the training ``cfg``
    dict (which in turn records ``base_model_path``).  You only need to supply
    ``base_model_path`` explicitly when the path stored in the checkpoint is no
    longer valid (e.g. the files were moved).

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint saved by the residual training script
        (``best_residual_model.pt`` or ``last_residual_model.pt``).
    base_model_path : str, optional
        Override the base-model ``.pt`` path stored inside the checkpoint.
    map_location : str
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    freeze_base : bool
        If True (default), the base model parameters have ``requires_grad=False``.

    Returns
    -------
    CombinedResidualDynamicsModel
        Ready-to-use combined model.  Call ``model.set_z(z)`` before rollout.

    Example
    -------
    >>> combined = load_combined_residual_model("results/residual/best_residual_model.pt", "results/base_model.pt")
    >>> combined.set_z(torch.zeros(16))          # broadcast to all batch elements
    >>> rollout = RolloutModel(dyn_model=combined, integration_method="euler",
    ...                        Tp=0.01, compile=False)
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    cfg  = ckpt["cfg"]

    # Resolve base model path
    model_path = base_model_path or cfg.get("base_model_path")
    if model_path is None:
        raise ValueError(
            "No base_model_path found in checkpoint cfg and none was supplied."
        )
    base_model, _ = load_learned_model(model_path)
    base_model = base_model.to(map_location)

    # Re-build encoder & residual MLP with the same hyper-parameters
    from ldm.systems.mlp.residual_model_helpers import build_models
    encoder, residual_mlp = build_models_from_cfg(cfg)

    encoder.load_state_dict(ckpt["encoder"])
    residual_mlp.load_state_dict(ckpt["residual_mlp"])

    encoder     = encoder.to(map_location)
    residual_mlp = residual_mlp.to(map_location)

    combined = CombinedResidualDynamicsModel(
        base_model   = base_model,
        residual_mlp = residual_mlp,
        z_dim        = cfg["z_dim"],
        freeze_base  = freeze_base,
    )
    combined._encoder = encoder   # attach for convenience (e.g. encoding a history)
    return combined


if __name__ == "__main__":
    config_path = "/home/piotr/mpc/learning_dynamics_models/experiments/eagle/neural_tires.yml"
    dynamics_model, _ = load_learned_model(config_path)
    print("Loaded dynamics model:", dynamics_model)
