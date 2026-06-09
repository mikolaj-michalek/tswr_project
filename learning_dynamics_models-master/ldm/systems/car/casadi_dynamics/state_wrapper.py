# --- Helper to mimic the StateWrappers ---
class CasadiStateWrapper:
    def __init__(self, x_u):
        # x_u is expected to be [v_x, v_y, r, omega_r, omega_f, delta]
        self.v_x = x_u[0]
        self.v_y = x_u[1]
        self.r = x_u[2]
        self.omega_wheels_rear = x_u[3]
        self.omega_wheels_front = x_u[4]
        self.delta = x_u[5]