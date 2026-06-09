import torch
import logging
from racing_env.utils.constants import CAR, STATE
from racing_env.utils.integrator import get_integrator
from racing_env.utils.metrics import Metrics

 
class BasicSimulator(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        dt: float = 0.05,
        frame_skip: int = 5,
        maximum_duration: float = 1.,
        integration_method: str = "rk4",
        num_envs: int = 1,
        randomize: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        logging.info("Initializing simulator")

         # Config
        self.device = device
        torch.set_default_device(self.device)
        self.timestep = dt / frame_skip
        self.frame_skip = frame_skip
        self.maximum_duration = maximum_duration
        self.num_envs = num_envs
        self.randomize = randomize

        self.state = torch.zeros((self.num_envs, STATE.SIZE), dtype=torch.float32, device=self.device)
        self.last_s = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.closest_idx = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self.eps = torch.tensor(1e-6, device=self.device)

        self.metrics = Metrics(num_envs=self.num_envs, device=self.device)

        self._integrate = get_integrator(integration_method)

        self.state_initializer = None

    def _calculate_observation(self):
        raise NotImplementedError(
            "This method should be implemented in the subclass. "
            "It should return the observation tensor based on the current self.state."
        )

    def _reset_history(self):
        raise NotImplementedError(
            "This method should be implemented in the subclass. "
            "It should reset any history that is being tracked."
        )

    def _randomize_params(self):
        raise NotImplementedError(
            "This method should be implemented in the subclass. "
            "It should randomize the parameters of the simulator."
        )

    def reset(self):
        """"
        Resets the simulator to the initial state and returns the initial observation.
        """
        info = {}
        self.metrics.reset()

        observation = self._reset_state()
        if self.randomize:
            self._randomize_params()
        return self.state, info, observation

    def _reset_state(self, mask=None, start_at_zero: bool = False):
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.state_initializer.initialize(mask, start_at_zero)
        self._reset_history(mask)
        return self._calculate_observation()

    def get_state(self):
        """
        Returns:
            torch.Tensor: Current state of the simulator [batch_size, state_dim]
        """
        return self.state

    def set_state(self, state):
        """
        Args:
            state (torch.Tensor): New state of the simulator [batch_size, state_dim]
        """
        self.state = state

    def pop_statistics(self):
        """
        Pops the current episode statistics and returns them.
        
        Returns:
            dict: Episode statistics including total reward, average reward,
            number of steps, progress, average speed, off track count,
            slightly off track count, and duration.
        """
        return self.metrics.pop_statistics()

    @property
    def dt(self) -> float:
        return self.timestep * self.frame_skip

    def set_s(self, s: float):
        self.last_s = torch.tensor(s, dtype=torch.float32, device=self.device)

    def get_s(self):
        return self.last_s.item()
    
    def get_x(self):
        return self.state[:, 0].item()
    
    def get_y(self):
        return self.state[:, 1].item()
    
    def get_yaw(self):
        return self.state[:, 2].item()

    def close(self):
        logging.info("Closing simulator")