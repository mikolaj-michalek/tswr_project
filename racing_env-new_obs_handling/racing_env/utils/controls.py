import torch

def nominal_controls_along_track(s, vx, track_s, track_curvature, car_length, dt, horizon: int = 1):
    dev = s.device
    num_envs = s.shape[0]
    times = torch.arange(horizon + 1, device=dev)[1:] * dt
    next_s = s[:, None] + times[None, :] * vx[:, None]
    s_dists = (track_s[:, None] - next_s[:, :, None]).abs()
    tracks_idxs = torch.argmin(s_dists, dim=2)
    curvatures = track_curvature[torch.arange(num_envs, device=dev), tracks_idxs]
    steering_angles = torch.atan(curvatures * car_length)
    currents = torch.zeros((num_envs, horizon), dtype=torch.float32, device=dev)
    controls = torch.stack([steering_angles, currents], dim=-1)
    return controls