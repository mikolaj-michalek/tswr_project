from racing_env.envs.initializers.interface import InitializerInterface
from racing_env.envs.initializers.random import RandomInitializer
from racing_env.envs.initializers.replay import ReplayInitializer
from racing_env.envs.initializers.zero import ZeroInitializer


def initializer_handler(initializer: str, simulator) -> InitializerInterface:
    """
    Factory function to create a state initializer based on the provided initializer type.
    
    Args:
        initializer (str): The type of initializer to create.  Supported values:
            * ``"zero"``   – :class:`ZeroInitializer` (random track position, zero controls).
            * ``"random"`` – :class:`RandomInitializer` (random position *and* velocities).
            * ``"replay"`` – :class:`ReplayInitializer` (mixed zero / replay-buffer init).
        simulator: The simulator instance to be used by the initializer.
    Returns:
        InitializerInterface: An instance of the specified state initializer.
    """
    if initializer == "zero":
        return ZeroInitializer(simulator)
    elif initializer == "random":
        return RandomInitializer(simulator)
    elif initializer.startswith("replay"):
        return ReplayInitializer(simulator, zero_fraction=1. - int(initializer.split("_")[1]) / 100.)
    else:
        raise ValueError(f"Unknown initializer type: {initializer}")