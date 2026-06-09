import torch
from racetrack import Racetrack


def load_tracks(tracks: list,
                num_envs: int,
                max_track_size: int,
                two_way_tracks: bool = True,
                device: torch.device = torch.device("cpu"),
                ):
    print(f"Loading tracks: {tracks}")
    numbers_of_tracks = len(tracks)
    envs_per_track = num_envs // numbers_of_tracks
    envs_per_track_rest = num_envs % numbers_of_tracks

    # Pre-allocate tensors
    tensor_shape = (int(num_envs), max_track_size)
    pad_value = 1e7

    track_x = torch.full(tensor_shape, pad_value, device=device)
    track_y = torch.full(tensor_shape, pad_value, device=device)
    track_heading = torch.full(tensor_shape, pad_value, device=device)
    track_width = torch.full(tensor_shape, pad_value, device=device)
    track_curvature = torch.full(tensor_shape, pad_value, device=device)
    track_s = torch.full(tensor_shape, pad_value, device=device)

    # Store original track lengths and track size
    track_lengths = torch.zeros((num_envs), dtype=torch.float32, device=device)
    track_size = torch.zeros((num_envs), dtype=torch.int64, device=device)

    def _fill_track_range(rt: Racetrack, start: int, count: int):
        """Load a single Racetrack and broadcast its data to `count` env rows at once."""
        if count <= 0:
            return
        end = start + count
        size = len(rt.x)

        # Convert each attribute to a tensor once, then broadcast across all rows.
        track_x[start:end, :size] = torch.tensor(rt.x, device=device).unsqueeze(0).expand(count, -1)
        track_y[start:end, :size] = torch.tensor(rt.y, device=device).unsqueeze(0).expand(count, -1)
        track_heading[start:end, :size] = torch.tensor(rt.heading, device=device).unsqueeze(0).expand(count, -1)
        track_width[start:end, :size] = torch.tensor(rt.track_width, device=device).unsqueeze(0).expand(count, -1)
        track_curvature[start:end, :size] = torch.tensor(rt.curvature, device=device).unsqueeze(0).expand(count, -1)
        track_s[start:end, :size] = torch.tensor(rt.s, device=device).unsqueeze(0).expand(count, -1)
        track_lengths[start:end] = rt.track_length
        track_size[start:end] = rt.track_size

    # create dict with track name and idx range name: int idx-start int idx-end
    tracks_idx = {}
    idx = 0

    for track in tracks:
        n_fwd = envs_per_track if not two_way_tracks else envs_per_track // 2
        n_flip = 0 if not two_way_tracks else envs_per_track - n_fwd
        tracks_idx[track] = (idx, idx + envs_per_track - 1)

        # Each unique track variant is instantiated and converted to tensors only once,
        # regardless of how many environments share it.
        _fill_track_range(Racetrack(track), idx, n_fwd)
        idx += n_fwd

        if two_way_tracks:
            _fill_track_range(Racetrack(track, flip=True), idx, n_flip)
            idx += n_flip

    if envs_per_track_rest > 0:
        _fill_track_range(Racetrack(tracks[0]), idx, envs_per_track_rest)
        idx += envs_per_track_rest

    return (numbers_of_tracks, track_lengths, track_size, track_x, track_y, track_heading,
            track_width, track_curvature, track_s, tracks_idx)
