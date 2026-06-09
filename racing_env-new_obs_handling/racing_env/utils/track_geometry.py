import torch
from racing_env.utils.debug.plot_track import plot_ego_transformed_boundaries
from racing_env.envs.simulators.robot_models.single_track_params import VehicleParameters
from racing_env.utils.constants import CAR

def get_ego_transformed_track_boundaries(x_ego, y_ego, yaw_ego, s, track_s,
                                         track_x, track_y, track_width,
                                         track_heading, foresight_spacing, foresight_size):
    """
    Transforms the next N points on the track to the ego vehicle's frame of reference.
    
    Returns:
        torch.Tensor: Points in ego frame, shape [batch_size, n_points, 4]
        Each point contains [x_left, y_left, x_right, y_right]
    """
    dev = x_ego.device
    batch_idx = torch.arange(x_ego.shape[0], device=dev)

    # Calculate rotation matrix components for each vehicle
    cos_yaw = torch.cos(yaw_ego)
    sin_yaw = -torch.sin(yaw_ego)

    next_s_norm = torch.linspace(0., 1., foresight_size + 1, device=dev)[None, 1:]
    next_s = s.unsqueeze(1) + next_s_norm * foresight_spacing * foresight_size

    # ignore placeholders
    max_track_s = torch.where(track_s > 1e6, torch.zeros_like(track_s), track_s).max(dim=1).values
    next_s = next_s % max_track_s.unsqueeze(1)

    idx = torch.searchsorted(track_s, next_s)

    ratio = (next_s - track_s[batch_idx.unsqueeze(1), idx - 1]) / (track_s[batch_idx.unsqueeze(1), idx] - track_s[batch_idx.unsqueeze(1), idx - 1])

    centerline_x = torch.lerp(track_x[batch_idx.unsqueeze(1), idx - 1], track_x[batch_idx.unsqueeze(1), idx], ratio)
    centerline_y = torch.lerp(track_y[batch_idx.unsqueeze(1), idx - 1], track_y[batch_idx.unsqueeze(1), idx], ratio)
    track_heading = torch.lerp(track_heading[batch_idx.unsqueeze(1), idx - 1], track_heading[batch_idx.unsqueeze(1), idx], ratio)
    track_width = torch.lerp(track_width[batch_idx.unsqueeze(1), idx - 1], track_width[batch_idx.unsqueeze(1), idx], ratio)
    
    # Calculate left and right boundary points using normal vectors
    normal_x = torch.sin(track_heading)
    normal_y = -torch.cos(track_heading)
    
    half_width = track_width / 2.0
    
    left_x = centerline_x + normal_x * half_width
    left_y = centerline_y + normal_y * half_width
    right_x = centerline_x - normal_x * half_width
    right_y = centerline_y - normal_y * half_width
    
    # Transform to ego frame
    # First translate
    left_x = left_x - x_ego.unsqueeze(1)
    left_y = left_y - y_ego.unsqueeze(1)
    right_x = right_x - x_ego.unsqueeze(1)
    right_y = right_y - y_ego.unsqueeze(1)
    
    # Then rotate
    left_x_ego = left_x * cos_yaw.unsqueeze(1) - left_y * sin_yaw.unsqueeze(1)
    left_y_ego = left_x * sin_yaw.unsqueeze(1) + left_y * cos_yaw.unsqueeze(1)
    right_x_ego = right_x * cos_yaw.unsqueeze(1) - right_y * sin_yaw.unsqueeze(1)
    right_y_ego = right_x * sin_yaw.unsqueeze(1) + right_y * cos_yaw.unsqueeze(1)
    
    # Stack left and right points together
    track_boundaries = torch.stack([
        left_x_ego, left_y_ego, right_x_ego, right_y_ego
    ], dim=2)

    
    # use for debugging
    #env_idx = 0
    #plot_ego_transformed_boundaries(torch.stack([x_ego, y_ego, yaw_ego], dim=-1)[env_idx],
    #                                track_x[env_idx, :track_size[env_idx]],
    #                                track_y[env_idx, :track_size[env_idx]],
    #                                track_boundaries[env_idx])
    
    return track_boundaries

@torch.compile(mode='max-autotune-no-cudagraphs')
def compute_position_on_track(pos, yaw, vehicle_params, track_x, track_y, track_width, track_size):
    """
    Computes the closest point on the track to the vehicle position.
    Per-corner collision uses actual world-space corner positions, each checked
    against the local track width at its own projection onto the centerline segment.

    Returns:
        closest_idx, p1_idx, p2_idx, t, signed_dist_to_centerline,
        signed_dist_to_edge, min_dist_to_edge, tire_dists_to_edge
    """
    vehicle_parameters = VehicleParameters()
    p = vehicle_parameters(vehicle_params)
    lf = p.lf
    lr = p.lr
    hw = CAR.WIDTH / 2.0

    batch_idx = torch.arange(pos.shape[0], device=pos.device)

    # --- Find closest track segment to car center ---
    track_points = torch.stack([track_x, track_y], dim=-1)  # [batch, track_size, 2]
    distances = torch.square(pos.unsqueeze(-2) - track_points).sum(-1)

    closest_idx = torch.argmin(distances, dim=-1)
    next_idx = (closest_idx + 1) % track_size
    prev_idx = (closest_idx - 1) % track_size
    is_next = torch.lt(distances[batch_idx, next_idx], distances[batch_idx, prev_idx])
    is_next = torch.where(closest_idx == 0, torch.ones_like(is_next, dtype=torch.bool), is_next)
    p1_idx = torch.where(is_next, closest_idx, prev_idx)
    p2_idx = torch.where(is_next, next_idx, closest_idx)

    # Segment endpoints
    p1_x = track_x[batch_idx, p1_idx]
    p1_y = track_y[batch_idx, p1_idx]
    p2_x = track_x[batch_idx, p2_idx]
    p2_y = track_y[batch_idx, p2_idx]

    track_segment_vec = torch.stack([p2_x - p1_x, p2_y - p1_y], dim=1)
    track_pos_vec = torch.stack([pos[:, 0] - p1_x, pos[:, 1] - p1_y], dim=1)

    segment_len_sq = torch.sum(track_segment_vec * track_segment_vec, dim=-1) + 1e-10
    t = torch.clamp(torch.sum(track_pos_vec * track_segment_vec, dim=-1) / segment_len_sq, 0.0, 1.0)

    closest_point = torch.stack([
        (1 - t) * p1_x + t * p2_x,
        (1 - t) * p1_y + t * p2_y,
    ], dim=-1)  # [batch, 2]

    cross = track_pos_vec[:, 0] * track_segment_vec[:, 1] - track_pos_vec[:, 1] * track_segment_vec[:, 0]
    centerline_side = torch.sign(cross)
    dist_to_centerline = torch.norm(pos - closest_point, dim=-1)
    signed_dist_to_centerline = centerline_side * dist_to_centerline

    # Track width at center projection
    tw_p1 = track_width[batch_idx, p1_idx]
    tw_p2 = track_width[batch_idx, p2_idx]
    center_track_width = torch.lerp(tw_p1, tw_p2, t)
    signed_dist_to_edge = center_track_width / 2 - dist_to_centerline

    # --- Per-corner collision: local search around center (O(B*4*W) vs O(B*4*T)) ---
    c_yaw = torch.cos(yaw)
    s_yaw = torch.sin(yaw)
    fwd_x, fwd_y = c_yaw * lf, s_yaw * lf
    bwd_x, bwd_y = c_yaw * lr, s_yaw * lr
    left_x, left_y = -s_yaw * hw, c_yaw * hw

    corners = torch.stack([
        torch.stack([pos[:, 0] + fwd_x + left_x, pos[:, 1] + fwd_y + left_y], dim=-1),  # FL
        torch.stack([pos[:, 0] + fwd_x - left_x, pos[:, 1] + fwd_y - left_y], dim=-1),  # FR
        torch.stack([pos[:, 0] - bwd_x + left_x, pos[:, 1] - bwd_y + left_y], dim=-1),  # RL
        torch.stack([pos[:, 0] - bwd_x - left_x, pos[:, 1] - bwd_y - left_y], dim=-1),  # RR
    ], dim=1)  # [batch, 4, 2]

    B = pos.shape[0]
    flat_corners = corners.reshape(B * 4, 2)
    c_center = closest_idx.repeat_interleave(4)
    c_track_x = track_x.repeat_interleave(4, dim=0)
    c_track_y = track_y.repeat_interleave(4, dim=0)
    c_track_w = track_width.repeat_interleave(4, dim=0)
    c_track_size = track_size.repeat_interleave(4)
    c_idx = torch.arange(B * 4, device=pos.device)

    # Local window: ±15 points (~1.5m) covers car length 0.43m
    c_w = 15
    c_offsets = torch.arange(-c_w, c_w + 1, device=pos.device)
    c_local_idx = (c_center.unsqueeze(1) + c_offsets) % c_track_size.unsqueeze(1)
    c_local_x = torch.gather(c_track_x, 1, c_local_idx)
    c_local_y = torch.gather(c_track_y, 1, c_local_idx)
    c_dists = torch.square(flat_corners.unsqueeze(1) - torch.stack([c_local_x, c_local_y], dim=-1)).sum(-1)
    c_local_argmin = torch.argmin(c_dists, dim=-1)
    c_closest = (c_center + c_offsets[c_local_argmin]) % c_track_size

    c_next = (c_closest + 1) % c_track_size
    c_prev = (c_closest - 1) % c_track_size
    c_next_pt = torch.stack([c_track_x[c_idx, c_next], c_track_y[c_idx, c_next]], dim=-1)
    c_prev_pt = torch.stack([c_track_x[c_idx, c_prev], c_track_y[c_idx, c_prev]], dim=-1)
    c_is_next = (flat_corners - c_next_pt).square().sum(-1) < (flat_corners - c_prev_pt).square().sum(-1)
    c_is_next = torch.where(c_closest == 0, torch.ones_like(c_is_next, dtype=torch.bool), c_is_next)
    c_p1 = torch.where(c_is_next, c_closest, c_prev)
    c_p2 = torch.where(c_is_next, c_next, c_closest)

    c_p1_x, c_p1_y = c_track_x[c_idx, c_p1], c_track_y[c_idx, c_p1]
    c_p2_x, c_p2_y = c_track_x[c_idx, c_p2], c_track_y[c_idx, c_p2]

    c_seg = torch.stack([c_p2_x - c_p1_x, c_p2_y - c_p1_y], dim=1)
    c_vec = torch.stack([flat_corners[:, 0] - c_p1_x, flat_corners[:, 1] - c_p1_y], dim=1)
    c_seg_len_sq = (c_seg * c_seg).sum(-1) + 1e-10
    c_t = torch.clamp((c_vec * c_seg).sum(-1) / c_seg_len_sq, 0.0, 1.0)

    c_proj = torch.stack([
        (1 - c_t) * c_p1_x + c_t * c_p2_x,
        (1 - c_t) * c_p1_y + c_t * c_p2_y,
    ], dim=-1)

    corner_dist = torch.norm(flat_corners - c_proj, dim=-1)
    corner_half_w = torch.lerp(c_track_w[c_idx, c_p1], c_track_w[c_idx, c_p2], c_t) / 2

    tire_dists_to_edge = (corner_half_w - corner_dist).reshape(B, 4)
    min_dist_to_edge = tire_dists_to_edge.min(dim=-1).values

    return closest_idx, p1_idx, p2_idx, t, signed_dist_to_centerline, signed_dist_to_edge, min_dist_to_edge, tire_dists_to_edge


def _segment_friction(pos: torch.Tensor, p1_idx: torch.Tensor, p2_idx: torch.Tensor,
                      track_x: torch.Tensor, track_y: torch.Tensor, friction: torch.Tensor,
                      batch_idx: torch.Tensor) -> torch.Tensor:
    """Project pos onto segment (p1_idx, p2_idx) and return interpolated friction."""
    p1_x = track_x[batch_idx, p1_idx]
    p1_y = track_y[batch_idx, p1_idx]
    p2_x = track_x[batch_idx, p2_idx]
    p2_y = track_y[batch_idx, p2_idx]
    seg = torch.stack([p2_x - p1_x, p2_y - p1_y], dim=1)
    vec = torch.stack([pos[:, 0] - p1_x, pos[:, 1] - p1_y], dim=1)
    seg_len_sq = (seg * seg).sum(-1) + 1e-10
    t = torch.clamp((vec * seg).sum(-1) / seg_len_sq, 0.0, 1.0)
    return torch.lerp(friction[batch_idx, p1_idx], friction[batch_idx, p2_idx], t)


def get_friction_at_positions(pos: torch.Tensor, track_x: torch.Tensor, track_y: torch.Tensor,
                              track_size: torch.Tensor, friction: torch.Tensor,
                              center_idx: torch.Tensor | None = None, window: int = 20) -> torch.Tensor:
    """
    Returns interpolated friction at pos [batch, 2].
    If center_idx is provided, searches only within ±window of center (O(batch*window) vs O(batch*track_size)).
    """
    batch_idx = torch.arange(pos.shape[0], device=pos.device)
    if center_idx is not None:
        w = window
        offsets = torch.arange(-w, w + 1, device=pos.device)
        local_idx = (center_idx.unsqueeze(1) + offsets) % track_size.unsqueeze(1)
        local_x = torch.gather(track_x, 1, local_idx)
        local_y = torch.gather(track_y, 1, local_idx)
        dists = torch.square(pos.unsqueeze(1) - torch.stack([local_x, local_y], dim=-1)).sum(-1)
        local_argmin = torch.argmin(dists, dim=-1)
        closest_idx = (center_idx + offsets[local_argmin]) % track_size
    else:
        track_pts = torch.stack([track_x, track_y], dim=-1)
        dists = torch.square(pos.unsqueeze(-2) - track_pts).sum(-1)
        closest_idx = torch.argmin(dists, dim=-1)

    next_idx = (closest_idx + 1) % track_size
    prev_idx = (closest_idx - 1) % track_size
    next_pt = torch.stack([track_x[batch_idx, next_idx], track_y[batch_idx, next_idx]], dim=-1)
    prev_pt = torch.stack([track_x[batch_idx, prev_idx], track_y[batch_idx, prev_idx]], dim=-1)
    is_next = (pos - next_pt).square().sum(-1) < (pos - prev_pt).square().sum(-1)
    is_next = torch.where(closest_idx == 0, torch.ones_like(is_next, dtype=torch.bool), is_next)
    p1_idx = torch.where(is_next, closest_idx, prev_idx)
    p2_idx = torch.where(is_next, next_idx, closest_idx)

    return _segment_friction(pos, p1_idx, p2_idx, track_x, track_y, friction, batch_idx)