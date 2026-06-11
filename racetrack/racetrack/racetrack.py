import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from racetrack.utils import compute_curvature, compute_heading, find_closest_neighbor_idx, find_closest_point_idx, find_projection
import torch


TRACKS_DIR = os.path.join(os.path.dirname(__file__), "tracks")
LIST_OF_TRACKS = [track_file.split(".")[0] for track_file in os.listdir(TRACKS_DIR)]

class Racetrack:
    f"""
    This class represents a racetrack.
    By now the racetrack can be chosen from the following options:
    {LIST_OF_TRACKS}
    """
    def __init__(self,
                 track_name: str,
                 flip: bool = False,
                 reverse: bool = False,
                 force_zero_tangent_coordinates: bool = False,
                 smoothing_factor: float = 0.3,
                 smoothing_spline_degree: int = 5,
                 smoothing_filter_window: int = 10,
                 smoothing_filter_order: int = 3,
                 interpolated_points_per_meter: int = 10
                 ):
        assert track_name in LIST_OF_TRACKS, f"Track {track_name} not found. Available tracks: {LIST_OF_TRACKS}"
        # load track
        self.track_name = track_name
        track_path = os.path.join(TRACKS_DIR, track_name + ".csv")
        data = np.loadtxt(track_path, delimiter=",")
        # TODO check if this makes sense with guys
        if not reverse:
            data = np.flip(data, axis=0)

        self.raw_x, self.raw_y = data[:, 0], data[:, 1]
        self.raw_track_width = data[:, 2] + data[:, 3]

        # force the track to touch the axes of the coordinate system
        if force_zero_tangent_coordinates:
            self.raw_x -= np.min(self.raw_x)
            self.raw_y -= np.min(self.raw_y)

        # TODO check if this makes sense with guys
        if flip:
            self.raw_y = -self.raw_y

        # spline smoothing factor in scipy.interpolate.splprep
        self.smoothing_factor = smoothing_factor
        self.smoothing_spline_degree = smoothing_spline_degree
        self.smoothing_filter_window = smoothing_filter_window
        self.smoothing_filter_order = smoothing_filter_order


        (
            self.s_smoothed, self.x_smoothed, self.y_smoothed, self.track_width_smoothed,
            self.track_width_corrected_smoothed, self.track_length_smoothed,
            self.curvature_smoothed, self.heading_smoothed,
            self.tck, self.u, self.x_dot_smoothed, self.y_dot_smoothed
        ) = self.compute_track_parameters(self.raw_x, self.raw_y, self.raw_track_width)

        self.heading_smoothed = np.unwrap(self.heading_smoothed)

        (
            self.s, self.x, self.y, self.track_width, self.track_width_corrected, self.track_length,
            self.curvature, self.heading,
        ) = self.interpolate_track(points_per_meter=interpolated_points_per_meter)
        self.track_size = len(self.x)
        
        # For track with overlap
        self.last_idxmindist = None
        if self.track_name == "long_track":
            self.find_closest_point_idx_fn = self.find_closest_point_idx_near_last
        else:
            self.find_closest_point_idx_fn = find_closest_point_idx

    def compute_track_parameters(self, raw_x, raw_y, raw_track_width):
        track_width = scipy.signal.savgol_filter(raw_track_width,
                                          self.smoothing_filter_window,
                                          self.smoothing_filter_order,
                                          mode="wrap")

        smoothing_weights = 1. / track_width
        tck, u = scipy.interpolate.splprep(
            [raw_x, raw_y], smoothing_weights, s=self.smoothing_factor, k=self.smoothing_spline_degree, per=len(raw_x)
        )

        # compute smoothed track parameters
        x, y = scipy.interpolate.splev(u, tck)
        x_dot, y_dot = scipy.interpolate.splev(u, tck, der=1)
        x_ddot, y_ddot = scipy.interpolate.splev(u, tck, der=2)
        heading = compute_heading(x_dot, y_dot)
        curvature = compute_curvature(x_dot, y_dot, x_ddot, y_ddot)
        s = scipy.integrate.cumulative_trapezoid(np.sqrt(x_dot**2 + y_dot**2), u, initial=0)
        track_length = s[-1]

        smoothing_errors = np.sqrt((x - raw_x) ** 2 + (y - raw_y) ** 2)
        track_width_corrected = (track_width / 2 - smoothing_errors) * 2 # TODO what is the purpose of this?
        # TODO check if s and u are the same
        return s, x, y, track_width, track_width_corrected, track_length, curvature, heading, tck, u, x_dot, y_dot

    def interpolate_track(self, points_per_meter=10):
        N = int(self.track_length_smoothed * points_per_meter)
        s_interpolated = np.linspace(0, self.track_length_smoothed, N)
        x_interpolated = np.interp(s_interpolated, self.s_smoothed, self.x_smoothed)
        y_interpolated = np.interp(s_interpolated, self.s_smoothed, self.y_smoothed)
        heading_interpolated = np.interp(s_interpolated, self.s_smoothed, self.heading_smoothed)
        curvature_interpolated = np.interp(s_interpolated, self.s_smoothed, self.curvature_smoothed)
        track_width_interpolated = np.interp(s_interpolated, self.s_smoothed, self.track_width_smoothed)
        track_width_corrected_interpolated = np.interp(s_interpolated, self.s_smoothed, self.track_width_corrected_smoothed)
        track_length_interpolated = s_interpolated[-1]

        return s_interpolated, x_interpolated, y_interpolated, track_width_interpolated, \
               track_width_corrected_interpolated, track_length_interpolated, \
               curvature_interpolated, heading_interpolated

    def debug_plot(self):
        # accumulate curvature to get heading angle
        heading_curv = scipy.integrate.cumulative_trapezoid(self.curvature, self.s, initial=0)
        heading_unwind = np.unwrap(self.heading)
        plt.figure()
        plt.subplot(5, 1, 1)
        plt.plot(self.x, self.x, "r", label="track points")
        # plot arrow for track direction
        plt.arrow(
            self.x[0],
            self.x[0],
            self.x[1],
            self.x[1],
            head_width=2,
            alpha=0.20,
            color="orange",
        )

        plt.subplot(5, 1, 2)
        plt.plot(self.x, heading_unwind - heading_unwind[0], label="heading")  
        plt.subplot(5, 1, 3)
        plt.plot(self.x, self.curvature, label="curvature")
        plt.subplot(5, 1, 4)
        plt.plot(self.x, heading_curv, label="heading_curv width")
        plt.subplot(5, 1, 5)
        plt.plot(self.x, self.x, label="x")
        plt.tight_layout()
        plt.show()

    def plot_track(self, plot_inport_dots=False, ax=None, n=None):
        print(f"Track length: {self.track_length_smoothed} m")
        print(f"{len(self.x_smoothed)} points")
        plt.rcParams["figure.figsize"] = (16, 8)
        plt.figure()
        # start arrow
        plt.arrow(
            self.x_smoothed[0],
            self.y_smoothed[0],
            self.x_smoothed[1] - self.x_smoothed[0],
            self.y_smoothed[1] - self.y_smoothed[0],
            head_width=2,
            alpha=0.20,
            color="orange",
        )

        if plot_inport_dots:
            plt.scatter(self.raw_x, self.raw_y)
        # print type x_s and y_s
        print("x shape: ", self.x_smoothed.shape, "y shape: ", self.y_smoothed.shape)
        plt.scatter(self.raw_x, self.raw_y, color="r", label="raw track points")

        plt.plot(self.x_smoothed, self.y_smoothed, "r", label="centerline spline")
        plt.title(f"Track map {self.track_name}", fontsize=20)

        for i in range(len(self.raw_x)):
            if plot_inport_dots:
                circle = plt.Circle(
                    (self.raw_x[i], self.raw_y[i]),
                    self.track_width_smoothed[i] / 2,
                    color="g",
                    fill=False,
                    alpha=0.15,
                )
                plt.gcf().gca().add_artist(circle)

            circle_2 = plt.Circle(
                (self.raw_x[i], self.raw_y[i]),
                self.track_width_corrected_smoothed[i] / 2,
                color="b",
                fill=False,
                alpha=0.15,
            )
            plt.gcf().gca().add_artist(circle_2)

        for i in range(0, len(self.x_smoothed), 20):
            plt.text(self.x_smoothed[i], self.y_smoothed[i] + 0.2, f"{self.s_smoothed[i]:.0f} ", fontsize=12)

        plt.grid()
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.legend(["centerline spline", "track direction"])
        
    def frenet2cart(self, s: np.ndarray, n: np.ndarray, mu: np.ndarray = None):
        s = np.mod(s, self.track_length_smoothed)   
        x_dot = np.interp(s, self.s_smoothed, self.x_dot_smoothed)
        y_dot = np.interp(s, self.s_smoothed, self.y_dot_smoothed)
        x = np.interp(s, self.s_smoothed, self.x_smoothed)
        y = np.interp(s, self.s_smoothed, self.y_smoothed)
        m = np.hypot(x_dot, y_dot)
        N = np.array([-y_dot, x_dot]) / m
        x_out = x + n * N[0]
        y_out = y + n * N[1]
        
        if mu is None:
            return x_out, y_out
        
        heading = np.interp(s, self.s_smoothed, self.heading_smoothed) + mu
        return x_out, y_out, heading

    def plot_points(
        self, s: np.ndarray, n: np.ndarray, v_x=None, marker="o", alpha=1.0
    ):
        x_out, y_out = self.frenet2cart(s, n)
        if v_x is not None:
            plt.scatter(x_out, y_out, c=v_x, cmap="jet", marker=marker, alpha=alpha)
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x_out, y_out, color="r", marker=marker, alpha=alpha)

    def plot_points_cartesian(self, x, y, v_x=None):
        if v_x is not None:
            plt.scatter(x, y, c=v_x, cmap="jet")
            plt.colorbar().set_label("v_x [m/s]")
        else:
            plt.scatter(x, y, color="r")

    def length(self):
        return self.track_length_smoothed

    def plot_curvature(self):
        plt.figure()
        plt.plot(self.s_smoothed, self.curvature_smoothed, label="curvature")
        plt.plot(self.s_smoothed, self.raw_track_width, label="width")
        plt.xlabel("s [m]")
        plt.ylabel("curvature [1/m]")
        plt.title("Curvature")
        plt.grid()
        plt.legend()
        plt.tight_layout()

    def plot_max_width(self):
        plt.figure()
        plt.plot(self.s_smoothed, self.raw_track_width, label="width")
        inv_k = 1 / np.abs(self.curvature_smoothed)
        inv_k = np.clip(inv_k, 0, 2)
        plt.plot(self.s_smoothed, inv_k, label="1/curvature")
        plt.xlabel("s [m]")
        plt.ylabel("width [m]")
        plt.title("Track width")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
    def find_closest_point_idx_near_last(self, x, y, x_ref, y_ref):
        distances = np.hypot(x - x_ref, y - y_ref)
        
        if self.last_idxmindist is not None:
            n_points = len(x_ref)
            window_size = 20  # Search window size
            mask = np.ones(n_points, dtype=bool)
            indices = np.arange(self.last_idxmindist - window_size,
                                self.last_idxmindist + window_size + 1)
            indices = indices % n_points
            mask[indices] = False
            distances[mask] = 1000.0
            
        idxmindist = np.argmin(distances)
        
        self.last_idxmindist = idxmindist
        
        # auto reset if the closest point is too far away
        # if distances[idxmindist] > 1.0:
        #     self.last_idxmindist = None
        
        return idxmindist
        

    def cart2frenet(self, heading, x, y):

        closest_point_idx = self.find_closest_point_idx_fn(x, y, self.x_smoothed, self.y_smoothed)
        closest_neighbor_idx = find_closest_neighbor_idx(x, y, self.x_smoothed, self.y_smoothed, closest_point_idx)

        t = find_projection(x, y, self.x_smoothed, self.y_smoothed, self.s_smoothed, closest_point_idx, closest_neighbor_idx)
        s0 = (1-t)*self.s_smoothed[closest_point_idx] + t*self.s_smoothed[closest_neighbor_idx]
        x0 = (1-t)*self.x_smoothed[closest_point_idx] + t*self.x_smoothed[closest_neighbor_idx]
        y0 = (1-t)*self.y_smoothed[closest_point_idx] + t*self.y_smoothed[closest_neighbor_idx]
        heading_1 = self.heading_smoothed[closest_point_idx]
        heading_2 = self.heading_smoothed[closest_neighbor_idx]
        if np.abs(heading_2 - heading_1) > np.pi:
            heading_2 -= 2 * np.pi * np.sign(heading_2)
        psi0 = (1-t)*heading_1 + t*heading_2
        
        s = s0
        n = np.cos(psi0) * (y - y0) - np.sin(psi0) * (x - x0)    
        alpha = heading - psi0
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        s = np.abs(s)
        return np.array([s, n, alpha])

    def compute_position_on_track(self, yaw, x, y, lf=0.2, lr=0.13, car_width=0.3):
        """
        Computes the closest point on the track to the vehicle position.
        Returns:
            closest_idx: Indices of the closest points on the track.
            dist_to_centerline: Distance to the centerline of the track.
            dist_to_edge: Distance to the edge of the track.
        """
        closest_point_idx = find_closest_point_idx(x, y, self.x_smoothed, self.y_smoothed)
        closest_neighbor_idx = find_closest_neighbor_idx(x, y, self.x_smoothed, self.y_smoothed, closest_point_idx)

        t = find_projection(x, y, self.x_smoothed, self.y_smoothed, self.s_smoothed, closest_point_idx, closest_neighbor_idx)
        progress = (1-t)*self.s_smoothed[closest_point_idx] + t*self.s_smoothed[closest_neighbor_idx]
        centerline_x = (1-t)*self.x_smoothed[closest_point_idx] + t*self.x_smoothed[closest_neighbor_idx]
        centerline_y = (1-t)*self.y_smoothed[closest_point_idx] + t*self.y_smoothed[closest_neighbor_idx]
        heading_1 = self.heading_smoothed[closest_point_idx]
        heading_2 = self.heading_smoothed[closest_neighbor_idx]
        if np.abs(heading_2 - heading_1) > np.pi:
            heading_2 -= 2 * np.pi * np.sign(heading_2)
            
        centerline_yaw = (1-t)*heading_1 + t*heading_2
        track_width = (1-t)*self.track_width_smoothed[closest_point_idx] + t*self.track_width_smoothed[closest_neighbor_idx]

        signed_dist_to_centerline = np.cos(centerline_yaw) * (y - centerline_y) - np.sin(centerline_yaw) * (x - centerline_x)    
        dist_to_centerline = np.abs(signed_dist_to_centerline)
        heading_error = yaw - centerline_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        p1_idx = np.where(closest_neighbor_idx > closest_point_idx, closest_point_idx, closest_neighbor_idx)
        p2_idx = np.where(closest_neighbor_idx > closest_point_idx, closest_neighbor_idx, closest_point_idx)

        hw = car_width / 2.0
        lat_w = hw * np.cos(heading_error)
        
        s_mu = np.sin(heading_error)
        lat_f = lf * s_mu
        lat_r = lr * s_mu

        n = signed_dist_to_centerline

        d_fl = n + lat_f + lat_w
        d_rl = n - lat_r + lat_w
        d_fr = n + lat_f - lat_w
        d_rr = n - lat_r - lat_w

        # Find the corner that is furthest from the center (largest absolute distance)
        all_corner_dists_from_center = np.stack([np.abs(d_fl), np.abs(d_fr), np.abs(d_rl), np.abs(d_rr)], axis=-1)
        max_dist_from_center = np.max(all_corner_dists_from_center, axis=-1)

        # The minimum distance to the edge is the Half-Width minus the furthest corner's distance
        min_dist_to_edge = (track_width / 2) - max_dist_from_center
        signed_dist_to_edge = (track_width / 2) - dist_to_centerline

        return progress, signed_dist_to_centerline, heading_error, closest_point_idx, p1_idx, p2_idx, t, signed_dist_to_edge, min_dist_to_edge