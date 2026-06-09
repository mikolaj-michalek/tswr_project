from typing import Optional

from tester.friction_profiles import get_friction_profile
from racing_env.envs.gymnasium import SingleTrackVecEnv


def apply_train_friction_profile(envs: SingleTrackVecEnv, profile_name: Optional[str]) -> None:
    """
    Apply a named friction profile to all environments in a SingleTrackVecEnv.

    If profile_name is falsy (\"\" or None), this is a no-op.
    """
    if not profile_name:
        return

    friction_spline_y, friction_spline_x = get_friction_profile(profile_name)
    envs.set_friction_curve_all(friction_spline_x, friction_spline_y)

