# ------------------------------------------------------------------------------
#                                   PHASE 6
# ------------------------------------------------------------------------------

import math
import torch


class Drift360Reward:
    """
    Reward do driftu 360.

    Fazy:
    - DRIVE: rozpędź się i trzymaj tor
    - DRIFT: wykonaj pierwszy obrót 360
    - RECOVER: wyprostuj auto po obrocie
    - FINISH: po pierwszym 360 jedź już tylko po torze do końca epizodu
    """

    DRIVE = 0
    DRIFT = 1
    RECOVER = 2
    FINISH = 3

    PHASE_NAMES = {
        DRIVE: "DRIVE",
        DRIFT: "DRIFT",
        RECOVER: "RECOVER",
        FINISH: "FINISH",
    }

    def __init__(
        self,
        env,
        yaw_target=2.0 * math.pi,
        min_start_speed=4.0,
        min_drift_speed=2.0,
        max_drift_steps=260,
        max_recover_steps=100,
        curvature_threshold=0.25,
        debug=True,
    ):
        self.env = env
        self.yaw_target = yaw_target
        self.min_start_speed = min_start_speed
        self.min_drift_speed = min_drift_speed
        self.max_drift_steps = max_drift_steps
        self.max_recover_steps = max_recover_steps
        self.curvature_threshold = curvature_threshold
        self.debug = debug

        n = env.num_envs
        device = env.device

        self.prev_yaw = torch.zeros(n, device=device)
        self.yaw_sum = torch.zeros(n, device=device)
        self.prev_yaw_abs = torch.zeros(n, device=device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)

        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.drift_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.recover_steps = torch.zeros(n, dtype=torch.long, device=device)

        self.best_yaw_abs = torch.zeros(n, device=device)
        self.finished_once = torch.zeros(n, dtype=torch.bool, device=device)

        self.last_debug_info = {}

    def _reset_internal_state(self):
        if not hasattr(self.env, "steps"):
            return

        reset = self.env.steps <= 1

        self.prev_yaw = torch.where(reset, torch.zeros_like(self.prev_yaw), self.prev_yaw)
        self.yaw_sum = torch.where(reset, torch.zeros_like(self.yaw_sum), self.yaw_sum)
        self.prev_yaw_abs = torch.where(reset, torch.zeros_like(self.prev_yaw_abs), self.prev_yaw_abs)
        self.initialized = torch.where(reset, torch.zeros_like(self.initialized), self.initialized)

        self.phase = torch.where(reset, torch.zeros_like(self.phase), self.phase)
        self.drift_steps = torch.where(reset, torch.zeros_like(self.drift_steps), self.drift_steps)
        self.recover_steps = torch.where(reset, torch.zeros_like(self.recover_steps), self.recover_steps)

        self.best_yaw_abs = torch.where(reset, torch.zeros_like(self.best_yaw_abs), self.best_yaw_abs)
        self.finished_once = torch.where(reset, torch.zeros_like(self.finished_once), self.finished_once)

    def _straight_gate(self):
        batch_idx = self.env._batch_idx
        closest_idx = self.env.closest_idx
        curvature = torch.abs(self.env.track_curvature[batch_idx, closest_idx])
        return curvature < self.curvature_threshold

    def _save_debug_info(self, speed, beta, r, dyaw, progress, reward):
        self.last_debug_info = {
            "phase": self.phase.detach().cpu().tolist(),
            "phase_name": [self.PHASE_NAMES[int(p)] for p in self.phase.detach().cpu().tolist()],
            "yaw_deg": (torch.abs(self.yaw_sum) * 180.0 / math.pi).detach().cpu().tolist(),
            "best_yaw_deg": (self.best_yaw_abs * 180.0 / math.pi).detach().cpu().tolist(),
            "speed": speed.detach().cpu().tolist(),
            "beta_deg": (torch.abs(beta) * 180.0 / math.pi).detach().cpu().tolist(),
            "r": torch.abs(r).detach().cpu().tolist(),
            "dyaw_deg": (torch.abs(dyaw) * 180.0 / math.pi).detach().cpu().tolist(),
            "progress": progress.detach().cpu().tolist(),
            "reward": reward.detach().cpu().tolist(),
            "drift_steps": self.drift_steps.detach().cpu().tolist(),
            "recover_steps": self.recover_steps.detach().cpu().tolist(),
            "finished_once": self.finished_once.detach().cpu().tolist(),
        }

    def _debug_print(self):
        if not self.debug or not hasattr(self.env, "steps"):
            return

        step = int(self.env.steps[0].item())

        if step % 50 != 0:
            return

        info = self.last_debug_info

        print(
            f"[REWARD] step={step:04d} | "
            f"phase={info['phase_name'][0]} | "
            f"yaw={info['yaw_deg'][0]:.1f} deg | "
            f"best={info['best_yaw_deg'][0]:.1f} deg | "
            f"v={info['speed'][0]:.2f} | "
            f"beta={info['beta_deg'][0]:.1f} deg | "
            f"r={info['r'][0]:.2f} | "
            f"rew={info['reward'][0]:+.3f}"
        )

    def __call__(self):
        self._reset_internal_state()
        old_phase = self.phase.clone()

        state = self.env.state

        yaw = state[:, 2]
        v_x = state[:, 3]
        v_y = state[:, 4]
        r = state[:, 5]
        delta = state[:, 8]

        self.prev_yaw = torch.where(self.initialized, self.prev_yaw, yaw)
        self.initialized = torch.ones_like(self.initialized)

        dyaw = yaw - self.prev_yaw
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        self.prev_yaw = yaw.clone()

        speed = torch.sqrt(v_x ** 2 + v_y ** 2)
        beta = torch.atan2(v_y, torch.clamp(torch.abs(v_x), min=0.1))

        abs_beta = torch.abs(beta)
        abs_r = torch.abs(r)
        abs_dyaw = torch.abs(dyaw)
        abs_delta = torch.abs(delta)
        abs_center = torch.abs(self.env.signed_dist_to_centerline)
        abs_heading = torch.abs(self.env.heading_diff)

        progress = self.env.progress / self.env.dt
        straight = self._straight_gate()

        # ------------------------------------------------------------
        # DRIVE: rozpędzenie i trzymanie toru.
        # ------------------------------------------------------------
        drive_reward = (
            0.35 * progress
            + 0.18 * torch.clamp(speed - 2.0, 0.0, 8.0)
            - 0.20 * abs_center
            - 0.12 * abs_heading
        )

        start_drift = (
            (self.phase == self.DRIVE)
            & (~self.finished_once)
            & straight
            & (speed > self.min_start_speed)
        )

        self.phase = torch.where(
            start_drift,
            torch.ones_like(self.phase) * self.DRIFT,
            self.phase,
        )

        self.yaw_sum = torch.where(start_drift, torch.zeros_like(self.yaw_sum), self.yaw_sum)
        self.prev_yaw_abs = torch.where(start_drift, torch.zeros_like(self.prev_yaw_abs), self.prev_yaw_abs)
        self.drift_steps = torch.where(start_drift, torch.zeros_like(self.drift_steps), self.drift_steps)

        # ------------------------------------------------------------
        # DRIFT: wykonanie jednego pełnego obrotu 360.
        # ------------------------------------------------------------
        in_drift = self.phase == self.DRIFT

        self.prev_yaw_abs = torch.where(in_drift, torch.abs(self.yaw_sum), self.prev_yaw_abs)
        self.yaw_sum = torch.where(in_drift, self.yaw_sum + dyaw, self.yaw_sum)

        yaw_abs = torch.abs(self.yaw_sum)
        self.best_yaw_abs = torch.maximum(self.best_yaw_abs, yaw_abs)
        self.drift_steps = torch.where(in_drift, self.drift_steps + 1, self.drift_steps)

        yaw_delta_abs = torch.relu(yaw_abs - self.prev_yaw_abs)
        yaw_frac = torch.clamp(yaw_abs / self.yaw_target, 0.0, 1.0)

        yaw_reward = 30.0 * yaw_delta_abs * (1.0 + 2.0 * yaw_frac)

        beta_reward = 1.2 * torch.clamp(abs_beta - 0.08, 0.0, 0.7)
        rotation_reward = 0.45 * torch.clamp(abs_r, 0.0, 5.0)
        speed_keep_reward = 0.20 * torch.clamp(speed, 0.0, 8.0)

        drift_reward = (
            yaw_reward
            + beta_reward
            + rotation_reward
            + speed_keep_reward
            - 0.18 * abs_center
            - 0.015 * abs_delta
        )

        late_drift = in_drift & (yaw_frac > 0.55)

        strong_counter_reward = 0.25 * torch.clamp(
            -delta * r,
            0.0,
            2.0,
        )

        drift_reward = torch.where(
            late_drift,
            drift_reward + strong_counter_reward,
            drift_reward,
        )

        near_finish = in_drift & (yaw_frac > 0.85)

        pre_recover_countersteer_reward = 0.20 * torch.clamp(
            -delta * r,
            0.0,
            2.0,
        )

        drift_reward = torch.where(
            near_finish,
            drift_reward + pre_recover_countersteer_reward,
            drift_reward,
        )

        too_slow = in_drift & (self.drift_steps > 20) & (speed < self.min_drift_speed)
        drift_reward = torch.where(too_slow, drift_reward - 6.0, drift_reward)

        failed_drift = in_drift & (self.drift_steps > self.max_drift_steps)
        drift_reward = torch.where(failed_drift, drift_reward - 10.0, drift_reward)

        finished_drift = in_drift & (yaw_abs >= self.yaw_target)
        drift_reward = torch.where(finished_drift, drift_reward + 80.0, drift_reward)

        self.phase = torch.where(
            finished_drift,
            torch.ones_like(self.phase) * self.RECOVER,
            self.phase,
        )

        self.phase = torch.where(
            failed_drift | too_slow,
            torch.zeros_like(self.phase),
            self.phase,
        )

        self.recover_steps = torch.where(
            finished_drift,
            torch.zeros_like(self.recover_steps),
            self.recover_steps,
        )

        # ------------------------------------------------------------
        # RECOVER: wyprostowanie auta po pierwszym 360.
        # ------------------------------------------------------------
        in_recover = self.phase == self.RECOVER
        self.recover_steps = torch.where(in_recover, self.recover_steps + 1, self.recover_steps)

        countersteer_reward = 0.30 * torch.clamp(
            -delta * r,
            0.0,
            1.8,
        )

        spin_continue_penalty = 0.45 * abs_dyaw

        recover_reward = (
            0.80 * progress
            + 0.20 * torch.clamp(speed, 0.0, 8.0)
            + countersteer_reward
            - spin_continue_penalty
            - 0.90 * abs_heading
            - 0.55 * abs_r
            - 0.35 * abs_beta
            - 0.25 * abs_center
        )

        stable = (
            in_recover
            & (self.recover_steps > 15)
            & (abs_heading < 0.45)
            & (abs_r < 1.0)
            & (abs_beta < 0.25)
            & (speed > 2.0)
        )

        failed_recover = in_recover & (self.recover_steps > self.max_recover_steps)

        recover_reward = torch.where(stable, recover_reward + 35.0, recover_reward)
        recover_reward = torch.where(failed_recover, recover_reward - 8.0, recover_reward)

        self.finished_once = torch.where(
            stable,
            torch.ones_like(self.finished_once),
            self.finished_once,
        )

        self.phase = torch.where(
            stable,
            torch.ones_like(self.phase) * self.FINISH,
            self.phase,
        )

        self.phase = torch.where(
            failed_recover,
            torch.zeros_like(self.phase),
            self.phase,
        )

        reset = failed_drift | too_slow | failed_recover

        self.yaw_sum = torch.where(reset, torch.zeros_like(self.yaw_sum), self.yaw_sum)
        self.prev_yaw_abs = torch.where(reset, torch.zeros_like(self.prev_yaw_abs), self.prev_yaw_abs)
        self.drift_steps = torch.where(reset, torch.zeros_like(self.drift_steps), self.drift_steps)
        self.recover_steps = torch.where(reset, torch.zeros_like(self.recover_steps), self.recover_steps)

        # ------------------------------------------------------------
        # FINISH: po pierwszym 360 jedź po torze do końca epizodu.
        # ------------------------------------------------------------
        in_finish = self.phase == self.FINISH

        finish_reward = (
            1.00 * progress
            + 0.25 * torch.clamp(speed, 0.0, 8.0)
            - 0.90 * abs_heading
            - 0.40 * abs_center
            - 0.45 * abs_beta
            - 0.35 * abs_r
            - 0.08 * abs_delta
        )

        # ------------------------------------------------------------
        # Wybór rewardu według fazy.
        # ------------------------------------------------------------
        reward = torch.where(in_drift, drift_reward, drive_reward)
        reward = torch.where(in_recover, recover_reward, reward)
        reward = torch.where(in_finish, finish_reward, reward)

        reward = torch.where(self.env.slightly_off_track, reward - 2.0, reward)
        reward = torch.where(self.env.off_track, torch.ones_like(reward) * -30.0, reward)

        reward = reward / 10.0

        self._save_debug_info(speed, beta, r, dyaw, progress, reward)
        self._debug_print()

        if self.debug and torch.any(old_phase != self.phase):
            for idx in torch.where(old_phase != self.phase)[0][:5]:
                i = int(idx.item())
                print(
                    f"[PHASE] env={i} "
                    f"{self.PHASE_NAMES[int(old_phase[i].item())]} -> "
                    f"{self.PHASE_NAMES[int(self.phase[i].item())]} | "
                    f"yaw={self.last_debug_info['yaw_deg'][i]:.1f} deg | "
                    f"best={self.last_debug_info['best_yaw_deg'][i]:.1f} deg | "
                    f"v={self.last_debug_info['speed'][i]:.2f} | "
                    f"beta={self.last_debug_info['beta_deg'][i]:.1f} deg"
                )

        return reward