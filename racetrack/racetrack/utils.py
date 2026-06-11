import numpy as np

def compute_heading(x_dot, y_dot):
    return np.arctan2(y_dot, x_dot)

def compute_curvature(x_dot, y_dot, x_ddot, y_ddot):
    return (x_dot * y_ddot - y_dot * x_ddot) / (x_dot**2 + y_dot**2)**1.5

def dist2D(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def find_closest_point_idx(x, y, x_ref, y_ref):
    distances = dist2D(x, x_ref, y, y_ref)
    closest_idx = np.argmin(distances)
    return closest_idx

def find_closest_neighbor_idx(x, y, x_ref, y_ref, closest_idx):
    idx_before = closest_idx - 1
    idx_after = (closest_idx + 1) % len(x_ref)
    dist_before = dist2D(x, x_ref[idx_before], y, y_ref[idx_before])
    dist_after = dist2D(x, x_ref[idx_after], y, y_ref[idx_after])
    return idx_before if dist_before < dist_after else idx_after

def find_projection(x, y, xref, yref, sref, closest_idx, closest_neighbor_idx):
    vabs = abs(sref[closest_idx] - sref[closest_neighbor_idx])
    vl = np.array([xref[closest_neighbor_idx] - xref[closest_idx],
                   yref[closest_neighbor_idx] - yref[closest_idx]])
    u = np.array([x - xref[closest_idx], y - yref[closest_idx]])
    t = np.dot(vl, u) / vabs**2
    return t


