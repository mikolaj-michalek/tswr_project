import numpy as np
import os


DST = "racetrack/racetrack/tracks/ml_wide_playground.csv"

# mały obwód centerline, duża szerokość robi plac
WIDTH = 12.0
HEIGHT = 12.0
RADIUS = 2.0

N_STRAIGHT = 35
N_ARC = 20

# realnie daje ogromny obszar wokół całej pętli
HALF_WIDTH = 18.0


def main():
    x0 = WIDTH / 2 - RADIUS
    y0 = HEIGHT / 2 - RADIUS

    x_top = np.linspace(-x0, x0, N_STRAIGHT, endpoint=False)
    y_top = np.ones_like(x_top) * HEIGHT / 2

    th = np.linspace(np.pi / 2, 0, N_ARC, endpoint=False)
    x_tr = x0 + RADIUS * np.cos(th)
    y_tr = y0 + RADIUS * np.sin(th)

    y_right = np.linspace(y0, -y0, N_STRAIGHT, endpoint=False)
    x_right = np.ones_like(y_right) * WIDTH / 2

    th = np.linspace(0, -np.pi / 2, N_ARC, endpoint=False)
    x_br = x0 + RADIUS * np.cos(th)
    y_br = -y0 + RADIUS * np.sin(th)

    x_bottom = np.linspace(x0, -x0, N_STRAIGHT, endpoint=False)
    y_bottom = np.ones_like(x_bottom) * (-HEIGHT / 2)

    th = np.linspace(-np.pi / 2, -np.pi, N_ARC, endpoint=False)
    x_bl = -x0 + RADIUS * np.cos(th)
    y_bl = -y0 + RADIUS * np.sin(th)

    y_left = np.linspace(-y0, y0, N_STRAIGHT, endpoint=False)
    x_left = np.ones_like(y_left) * (-WIDTH / 2)

    th = np.linspace(np.pi, np.pi / 2, N_ARC, endpoint=False)
    x_tl = -x0 + RADIUS * np.cos(th)
    y_tl = y0 + RADIUS * np.sin(th)

    x = np.concatenate([x_top, x_tr, x_right, x_br, x_bottom, x_bl, x_left, x_tl])
    y = np.concatenate([y_top, y_tr, y_right, y_br, y_bottom, y_bl, y_left, y_tl])

    half_width = np.ones_like(x) * HALF_WIDTH

    os.makedirs(os.path.dirname(DST), exist_ok=True)

    with open(DST, "w") as f:
        f.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
        for xi, yi, wi in zip(x, y, half_width):
            f.write(f"{xi:.4f},{yi:.4f},{wi:.4f},{wi:.4f}\n")

    print(f"Saved: {DST}")
    print(f"CSV points: {len(x)}")
    print(f"x range: {x.min():.2f} - {x.max():.2f}")
    print(f"y range: {y.min():.2f} - {y.max():.2f}")
    print(f"Half width: {HALF_WIDTH:.2f}")
    print(f"Full width: {2 * HALF_WIDTH:.2f}")


if __name__ == "__main__":
    main()