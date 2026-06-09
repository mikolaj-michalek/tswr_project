import torch
from racing_env.utils.delay_fifo import DelayBufferFIFO


class TimeDelayFirstOrder(torch.nn.Module):
    def __init__(self, dt, delay_time, tau, num_envs, device: torch.device = torch.device("cpu")):
        super(TimeDelayFirstOrder, self).__init__()

        self.dt = dt
        self.tau = tau
        self.delay_buffer = DelayBufferFIFO(int(delay_time / dt), num_envs, device=device)
        self.x_delayed = torch.zeros(num_envs, device=device)

    def forward(self, x):
        dxdt = (self.x_delayed - x) / self.tau
        return dxdt

    def update_delayed_input(self, u):
        self.x_delayed = self.delay_buffer(u)

    def copy(self):
        new = TimeDelayFirstOrder(
            self.dt, self.delay_buffer.buffor.shape[-1] * self.dt, self.tau,
            num_envs=self.x_delayed.shape[0], device=self.x_delayed.device)
        new.delay_buffer = self.delay_buffer.copy()
        new.x_delayed = self.x_delayed.clone()
        return new

    def get_internal_state(self):
        return [*self.delay_buffer.get_internal_state(), self.x_delayed]

    def set_internal_state(self, state):
        self.delay_buffer.set_internal_state(state[:-1])
        self.x_delayed = state[-1].clone()
