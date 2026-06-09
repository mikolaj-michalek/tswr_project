import torch


class RolloutModel(torch.nn.Module):
    def __init__(self,
                 dyn_model: torch.nn.Module,
                 integration_method: str,
                 compile: bool,
                 Tp: float,
                 state_extender: torch.nn.Module = None,
        ):

        super(RolloutModel, self).__init__()
        self.dyn_model = dyn_model
        self.state_extender = state_extender if state_extender is not None else (lambda H, X0: X0)
        self.Tp = Tp

        assert integration_method == "rk4" or integration_method == "euler"
        self.integration_method = integration_method

        if self.integration_method == "rk4":
            self._step = self._rk4_step
            print("Using RK4 integration method for rollout")
        else:
            self._step = self._euler_step

        if compile:
            self._forward_n_step = torch.compile(self._forward_n_step, fullgraph=True, mode='max-autotune-no-cudagraphs')
            #self._forward_n_step = torch.compile(self._forward_n_step, fullgraph=True, mode='max-autotune')
            #self._forward_n_step = torch.compile(self._forward_n_step, fullgraph=True, mode='default')

    def _euler_step(self, t, X_sim, U):
        dx, tire_froces, slips = self.dyn_model(t, X_sim.clone(), U)
        dx = torch.nn.functional.pad(dx, (0, X_sim.shape[-1] - dx.shape[-1]))
        return X_sim + dx * self.Tp, tire_froces, slips

    def _rk4_step(self, t, X_sim, U):
        k1, tf1, slips1 = self.dyn_model(t, X_sim.clone(), U)
        k2, tf2, slips2 = self.dyn_model(t, X_sim.clone() + k1 * self.Tp / 2, U)
        k3, tf3, slips3 = self.dyn_model(t, X_sim.clone() + k2 * self.Tp / 2, U)
        k4, tf4, slips4 = self.dyn_model(t, X_sim.clone() + k3 * self.Tp, U)
        tire_forces = (tf1 + 2. * tf2 + 2. * tf3 + tf4) / 6.
        slips = (slips1 + 2. * slips2 + 2. * slips3 + slips4) / 6.
        dx = (k1 + 2*k2 + 2*k3 + k4) / 6.
        dx = torch.nn.functional.pad(dx, (0, X_sim.shape[-1] - dx.shape[-1]))
        return X_sim + dx * self.Tp, tire_forces, slips

    #def forward_old(self, X0, U, prediction_horizon: int):
    #    """
    #        X0: (batch_size, 1, state_dim)
    #        U: (batch_size, prediction_horizon, control_dim)

    #        return: X_sim (batch_size, prediction_horizon, state_dim)
    #    """
    #    
    #    X_sim = X0.repeat(1, prediction_horizon, 1)
    #    for i in range(1, prediction_horizon):
    #        t_now = torch.tensor([i * self.Tp], device=X0.device)
    #        X_sim[:, i] = self._step(t_now,
    #                                X_sim[:, i - 1],
    #                                U[:, i-1],)
    #        
    #    return X_sim
    
    
    def _forward_n_step(self, X0, U, prediction_horizon: int):
        """
            X0: (batch_size, 1, state_dim)
            U: (batch_size, prediction_horizon, control_dim)
            return: X_sim (batch_size, prediction_horizon, state_dim)
        """
        X_return = torch.zeros(X0.shape[0], prediction_horizon, X0.shape[1], device=X0.device)
        tire_forces = 0.
        for i in range(1, prediction_horizon + 1):
            t_now = torch.tensor([i * self.Tp], device=X0.device)
            X_next, tire_forces_, slips_ = self._step(t_now, X0, U[:, i-1])       
            X_return[:, i - 1] = torch.clamp(X_next, min=torch.tensor([0.5] + [-torch.inf] * (X0.shape[-1] - 1)))  # v_x min clamp
            X0 = X_return[:, i - 1]
            tire_forces = tire_forces + tire_forces_
        return X_return, tire_forces


    def forward(self, H, X0, U, prediction_horizon: int, chunk_size: int = 1):
        """
        Forward method that uses _forward_n_step in overlapping chunks.

        Args:
            H: (batch_size, history_length, state_dim)
            X0: (batch_size, 1, state_dim)
            U: (batch_size, prediction_horizon, control_dim)
            prediction_horizon: int
            chunk_size: int

        Returns:
            X_sim: (batch_size, prediction_horizon + 1, state_dim)
        """
        assert prediction_horizon % chunk_size == 0, \
            f"prediction_horizon {prediction_horizon} must be divisible by {chunk_size}"
        num_chunks = prediction_horizon // chunk_size

        state_dim = X0.shape[2]

        X0 = self.state_extender(H, X0)

        # Initialize X_sim with zeros and set the first state
        X_sim = torch.zeros(X0.shape[0], prediction_horizon + 1, X0.shape[2], device=X0.device)
        X_sim[:, 0] = X0.squeeze(1)

        #tire_forces = 0.
        tire_forces = []

        for i in range(num_chunks):
            # Calculate start and end indices for the current chunk
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, prediction_horizon)  # Ensure we don't exceed the horizon
            current_n = end_idx - start_idx  # Adjust n for the last chunk if necessary

            # Handle the indexing for U
            U_chunk = U[:, start_idx:end_idx]
            X0_chunk = X_sim[:, start_idx]

            # Compute the state evolution for the chunk
            X_chunk, tire_forces_ = self._forward_n_step(X0_chunk, U_chunk, current_n)
            #X_chunk = self._forward_n_step(X0_chunk, U_chunk, current_n)
            #tire_forces = tire_forces + tire_forces_
            tire_forces.append(tire_forces_)

            # Assign the computed states to X_sim
            X_sim[:, start_idx + 1:end_idx + 1] = X_chunk

        tire_forces = torch.stack(tire_forces, dim=1)  # (batch_size, num_chunks, force_dim)
        return X_sim[:, 1:, :state_dim], tire_forces # Exclude the initial state