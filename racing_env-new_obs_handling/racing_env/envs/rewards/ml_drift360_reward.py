import math
import torch


class Drift360Reward:
    DRIVE = 0
    SPIN = 1
    RECOVER = 2

    def __init__(
        self,
        env,
        yaw_target=2.0 * math.pi,
        min_spin_speed=5.0,
        curvature_threshold=0.12,
        max_spin_steps=110,
        cooldown_steps=140,
    ):
        self.env = env
        self.yaw_target = yaw_target
        self.min_spin_speed = min_spin_speed
        self.curvature_threshold = curvature_threshold
        self.max_spin_steps = max_spin_steps
        self.cooldown_steps = cooldown_steps

        self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
        self.yaw_sum = torch.zeros(env.num_envs, device=env.device)
        self.prev_yaw_sum = torch.zeros(env.num_envs, device=env.device)

        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.spin_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.recover_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.cooldown = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def _reset_internal_state(self):
        if hasattr(self.env, "steps"):
            reset_mask = self.env.steps <= 1

            self.prev_yaw = torch.where(reset_mask, torch.zeros_like(self.prev_yaw), self.prev_yaw)
            self.yaw_sum = torch.where(reset_mask, torch.zeros_like(self.yaw_sum), self.yaw_sum)
            self.prev_yaw_sum = torch.where(reset_mask, torch.zeros_like(self.prev_yaw_sum), self.prev_yaw_sum)
            self.phase = torch.where(reset_mask, torch.zeros_like(self.phase), self.phase)
            self.initialized = torch.where(reset_mask, torch.zeros_like(self.initialized), self.initialized)
            self.spin_steps = torch.where(reset_mask, torch.zeros_like(self.spin_steps), self.spin_steps)
            self.recover_steps = torch.where(reset_mask, torch.zeros_like(self.recover_steps), self.recover_steps)
            self.cooldown = torch.where(reset_mask, torch.zeros_like(self.cooldown), self.cooldown)

    def _straight_gate(self):
        batch_idx = self.env._batch_idx
        closest_idx = self.env.closest_idx
        local_curvature = torch.abs(self.env.track_curvature[batch_idx, closest_idx])

        on_straight = local_curvature < self.curvature_threshold
        aligned = torch.abs(self.env.heading_diff) < 0.55

        return on_straight & aligned

    def __call__(self):
        self._reset_internal_state()

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

        abs_dyaw = torch.abs(dyaw)

        speed = torch.sqrt(v_x ** 2 + v_y ** 2)
        beta = torch.atan2(v_y, torch.clamp(torch.abs(v_x), min=0.1))
        abs_beta = torch.abs(beta)
        abs_r = torch.abs(r)

        progress = self.env.progress / self.env.dt
        straight = self._straight_gate()

        self.cooldown = torch.clamp(self.cooldown - 1, min=0)

        # ============================================================
        # DRIVE: jazda + rozpędzenie na prostej
        # ============================================================
        drive_reward = (
            1.00 * progress
            - 0.35 * torch.abs(self.env.signed_dist_to_centerline)
            - 0.35 * torch.abs(self.env.heading_diff)
            - 0.02 * torch.abs(delta)
        )

        # mocniej premiujemy przygotowanie prędkości na prostej
        straight_speed_bonus = torch.where(
            straight,
            0.75 * torch.clamp(speed - 3.0, 0.0, 8.0),
            torch.zeros_like(speed),
        )

        drive_reward = drive_reward + straight_speed_bonus

        can_start_spin = (
            (self.phase == self.DRIVE)
            & straight
            & (speed > self.min_spin_speed)
            & (self.cooldown == 0)
        )

        self.phase = torch.where(
            can_start_spin,
            torch.ones_like(self.phase) * self.SPIN,
            self.phase,
        )

        self.yaw_sum = torch.where(can_start_spin, torch.zeros_like(self.yaw_sum), self.yaw_sum)
        self.prev_yaw_sum = torch.where(can_start_spin, torch.zeros_like(self.prev_yaw_sum), self.prev_yaw_sum)
        self.spin_steps = torch.where(can_start_spin, torch.zeros_like(self.spin_steps), self.spin_steps)

        # ============================================================
        # SPIN: uproszczony reward — liczy się dokończenie obrotu
        # ============================================================
        in_spin = self.phase == self.SPIN

        self.prev_yaw_sum = torch.where(in_spin, self.yaw_sum, self.prev_yaw_sum)
        self.yaw_sum = torch.where(in_spin, self.yaw_sum + abs_dyaw, self.yaw_sum)
        self.spin_steps = torch.where(in_spin, self.spin_steps + 1, self.spin_steps)

        yaw_fraction = torch.clamp(self.yaw_sum / self.yaw_target, 0.0, 1.0)
        prev_yaw_fraction = torch.clamp(self.prev_yaw_sum / self.yaw_target, 0.0, 1.0)

        # mocno nieliniowy progres: 90 stopni mało warte, końcówka dużo warta
        yaw_stage_reward = 90.0 * (yaw_fraction ** 3 - prev_yaw_fraction ** 3)

        # milestone bonusy pomagają PPO zauważyć 180/270/360
        passed_180 = (prev_yaw_fraction < 0.50) & (yaw_fraction >= 0.50)
        passed_270 = (prev_yaw_fraction < 0.75) & (yaw_fraction >= 0.75)

        milestone_bonus = torch.zeros_like(speed)
        milestone_bonus = torch.where(passed_180, milestone_bonus + 12.0, milestone_bonus)
        milestone_bonus = torch.where(passed_270, milestone_bonus + 25.0, milestone_bonus)

        spin_reward = (
            yaw_stage_reward
            + milestone_bonus
            + 0.08 * progress
            + 0.12 * torch.clamp(speed, 0.0, 10.0)
            + 0.12 * torch.clamp(abs_beta, 0.0, 1.3)
            - 0.20 * torch.abs(self.env.signed_dist_to_centerline)
        )

        failed_spin = (
            in_spin
            & (
                (self.spin_steps > self.max_spin_steps)
                | ((speed < 2.0) & (self.spin_steps > 15))
                | ((self.yaw_sum < 0.35 * self.yaw_target) & (self.spin_steps > 45))
            )
        )

        finished_spin = in_spin & (self.yaw_sum >= self.yaw_target)

        spin_reward = torch.where(
            failed_spin,
            spin_reward - 55.0,
            spin_reward,
        )

        spin_reward = torch.where(
            finished_spin,
            spin_reward + 130.0,
            spin_reward,
        )

        self.phase = torch.where(
            finished_spin,
            torch.ones_like(self.phase) * self.RECOVER,
            self.phase,
        )

        self.phase = torch.where(
            failed_spin,
            torch.zeros_like(self.phase),
            self.phase,
        )

        self.cooldown = torch.where(
            failed_spin,
            torch.ones_like(self.cooldown) * self.cooldown_steps,
            self.cooldown,
        )

        self.recover_steps = torch.where(
            finished_spin,
            torch.zeros_like(self.recover_steps),
            self.recover_steps,
        )

        overspin = torch.nn.functional.relu(self.yaw_sum - 2.25 * math.pi)
        spin_reward = spin_reward - 5.0 * overspin

        # ============================================================
        # RECOVER: po 360 trzeba odzyskać kontrolę
        # ============================================================
        in_recover = self.phase == self.RECOVER

        self.recover_steps = torch.where(
            in_recover,
            self.recover_steps + 1,
            self.recover_steps,
        )

        recover_reward = (
            1.20 * progress
            + 0.20 * torch.clamp(speed, 0.0, 8.0)
            - 1.00 * torch.abs(self.env.heading_diff)
            - 0.65 * abs_r
            - 0.40 * abs_beta
            - 0.30 * torch.abs(self.env.signed_dist_to_centerline)
        )

        stable_after_spin = (
            in_recover
            & (self.recover_steps > 20)
            & (torch.abs(self.env.heading_diff) < 0.35)
            & (abs_r < 0.9)
            & (speed > 2.8)
        )

        recover_reward = torch.where(
            stable_after_spin,
            recover_reward + 50.0,
            recover_reward,
        )

        self.phase = torch.where(
            stable_after_spin,
            torch.zeros_like(self.phase),
            self.phase,
        )

        self.cooldown = torch.where(
            stable_after_spin,
            torch.ones_like(self.cooldown) * self.cooldown_steps,
            self.cooldown,
        )

        reset_spin_state = stable_after_spin | failed_spin

        self.yaw_sum = torch.where(reset_spin_state, torch.zeros_like(self.yaw_sum), self.yaw_sum)
        self.prev_yaw_sum = torch.where(reset_spin_state, torch.zeros_like(self.prev_yaw_sum), self.prev_yaw_sum)
        self.spin_steps = torch.where(reset_spin_state, torch.zeros_like(self.spin_steps), self.spin_steps)

        # ============================================================
        # wybór rewardu
        # ============================================================
        reward = torch.where(in_spin, spin_reward, drive_reward)
        reward = torch.where(in_recover, recover_reward, reward)

        slightly_off_track_penalty = (
            -5.0 * torch.nn.functional.relu(-self.env.signed_dist_to_edge + 0.05) ** 2.0
        )

        reward = torch.where(
            self.env.slightly_off_track,
            reward + slightly_off_track_penalty,
            reward,
        )

        reward = torch.where(
            self.env.off_track,
            torch.ones_like(reward) * -35.0,
            reward,
        )

        return reward / 10.0