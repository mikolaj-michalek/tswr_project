import torch

class RolloutModelWithHistory(torch.nn.Module):
    """
    Rollout model that maintains a sliding window of the N most recent
    observations and passes it to the dynamics model at every integration step.

    The dynamics model is expected to have the signature:
        dyn_model(t, history, X_sim, U) -> (dx, tire_forces, slips)
    where
        history : (batch_size, history_len, state_dim)  – N most recent states
        X_sim   : (batch_size, state_dim)               – current state
        U       : (batch_size, control_dim)             – current control

    Args:
        dyn_model          : dynamics model with the history-aware signature above
        integration_method : "euler" or "rk4"
        compile            : whether to torch.compile the inner rollout
        Tp                 : time-step length (seconds)
        history_len        : number of past observations to keep (N)
    """

    def __init__(
        self,
        dyn_model: torch.nn.Module,
        integration_method: str,
        compile: bool,
        Tp: float,
        history_len: int,
    ):
        super().__init__()
        self.dyn_model = dyn_model
        self.Tp = Tp
        self.history_len = history_len

        assert integration_method in ("rk4", "euler"), \
            f"Unknown integration method: {integration_method}"
        self.integration_method = integration_method

        if self.integration_method == "rk4":
            self._step = self._rk4_step
            print("Using RK4 integration method for RolloutModelWithHistory")
        else:
            self._step = self._euler_step

        if compile:
            self._forward_n_step = torch.compile(
                self._forward_n_step, fullgraph=True, mode="max-autotune-no-cudagraphs"
            )

    # ------------------------------------------------------------------
    # Low-level integration steps
    # ------------------------------------------------------------------

    def _euler_step(self, t, history, X_sim, U):
        dx, tire_forces, slips = self.dyn_model(t, history, X_sim.clone(), U)
        dx = torch.nn.functional.pad(dx, (0, X_sim.shape[-1] - dx.shape[-1]))
        return X_sim + dx * self.Tp, tire_forces, slips

    def _rk4_step(self, t, history, X_sim, U):
        k1, tf1, s1 = self.dyn_model(t, history, X_sim.clone(), U)
        k2, tf2, s2 = self.dyn_model(t, history, X_sim.clone() + k1 * self.Tp / 2, U)
        k3, tf3, s3 = self.dyn_model(t, history, X_sim.clone() + k2 * self.Tp / 2, U)
        k4, tf4, s4 = self.dyn_model(t, history, X_sim.clone() + k3 * self.Tp, U)
        tire_forces = (tf1 + 2.0 * tf2 + 2.0 * tf3 + tf4) / 6.0
        slips = (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        dx = torch.nn.functional.pad(dx, (0, X_sim.shape[-1] - dx.shape[-1]))
        return X_sim + dx * self.Tp, tire_forces, slips

    # ------------------------------------------------------------------
    # Inner n-step rollout (potentially compiled)
    # ------------------------------------------------------------------

    def _forward_n_step(self, history, X0, U, prediction_horizon: int):
        """
        Args:
            history           : (batch_size, history_len, state_dim)
            X0                : (batch_size, state_dim)
            U                 : (batch_size, prediction_horizon, control_dim)
            prediction_horizon: int

        Returns:
            X_return : (batch_size, prediction_horizon, state_dim)
            tire_forces : accumulated tire forces scalar / tensor
        """
        X_return = torch.zeros(
            X0.shape[0], prediction_horizon, X0.shape[-1], device=X0.device
        )
        tire_forces_acc = 0.0

        # Make a mutable copy of the history buffer
        hist = history.clone()  # (batch_size, history_len, state_dim)

        for i in range(1, prediction_horizon + 1):
            t_now = torch.tensor([i * self.Tp], device=X0.device)
            X_next, tf_, _ = self._step(t_now, hist, X0, U[:, i - 1])
            X_next = torch.clamp(
                X_next,
                min=torch.tensor([0.5] + [-torch.inf] * (X0.shape[-1] - 1)),
            )
            X_return[:, i - 1] = X_next
            tire_forces_acc = tire_forces_acc + tf_

            # Slide the history window: drop oldest, append current state
            hist = torch.cat([hist[:, 1:], torch.cat([X0, U[:, i - 1]], dim=-1).unsqueeze(1)], dim=1)
            X0 = X_next

        return X_return, tire_forces_acc

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, H, X0, U, prediction_horizon: int, chunk_size: int = 1):
        """
        Args:
            H                 : (batch_size, history_len, state_dim)  – observation history
            X0                : (batch_size, 1, state_dim)            – initial state
            U                 : (batch_size, prediction_horizon, control_dim)
            prediction_horizon: int
            chunk_size        : int  (prediction_horizon must be divisible by chunk_size)

        Returns:
            X_sim       : (batch_size, prediction_horizon, state_dim)
            tire_forces : (batch_size, num_chunks, force_dim)
        """
        assert prediction_horizon % chunk_size == 0, (
            f"prediction_horizon {prediction_horizon} must be divisible by chunk_size {chunk_size}"
        )
        num_chunks = prediction_horizon // chunk_size
        state_dim = X0.shape[2]

        # Pad or truncate history to exactly history_len steps
        if H.shape[1] < self.history_len:
            pad = X0.expand(-1, self.history_len - H.shape[1], -1)
            H = torch.cat([pad, H], dim=1)
        elif H.shape[1] > self.history_len:
            H = H[:, -self.history_len:]

        # Build the full state buffer (history + X0 as the most recent entry)
        # Shape: (batch_size, history_len, state_dim)
        #hist = torch.cat([H[:, 1:], X0], dim=1)  # slide X0 into the window

        X_sim = torch.zeros(
            X0.shape[0], prediction_horizon + 1, state_dim, device=X0.device
        )
        X_sim[:, 0] = X0.squeeze(1)

        tire_forces_list = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, prediction_horizon)
            current_n = end_idx - start_idx

            U_chunk = U[:, start_idx:end_idx]
            X0_chunk = X_sim[:, start_idx]

            # History for this chunk: last history_len (state, action) pairs
            # available before this chunk start.
            # Full sequence = original H  +  (X_sim[:,0], U[:,0]), ..., (X_sim[:,start_idx-1], U[:,start_idx-1])
            if start_idx == 0:
                hist_chunk = H
            else:
                xu_recent = torch.cat(
                    [X_sim[:, :start_idx, :state_dim], U[:, :start_idx]], dim=-1
                )  # (batch, start_idx, state_dim + control_dim)
                full_hist = torch.cat([H, xu_recent], dim=1)  # (batch, history_len + start_idx, obs_dim)
                hist_chunk = full_hist[:, -self.history_len:]  # (batch, history_len, obs_dim)

            X_chunk, tf_ = self._forward_n_step(hist_chunk, X0_chunk, U_chunk, current_n)
            tire_forces_list.append(tf_)
            X_sim[:, start_idx + 1:end_idx + 1] = X_chunk

        tire_forces = torch.stack(tire_forces_list, dim=1)  # (batch_size, num_chunks, ...)
        return X_sim[:, 1:, :state_dim], tire_forces
