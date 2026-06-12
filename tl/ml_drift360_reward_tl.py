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
            min_spin_speed=4.0,
            curvature_threshold=0.12,
    ):
        self.env = env
        self.yaw_target = yaw_target
        self.min_spin_speed = min_spin_speed
        self.curvature_threshold = curvature_threshold

        self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
        self.yaw_sum = torch.zeros(env.num_envs, device=env.device)
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.recover_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.cooldown = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def _reset_internal_state(self):
        if hasattr(self.env, "steps"):
            reset_mask = self.env.steps <= 1

            self.prev_yaw = torch.where(reset_mask, torch.zeros_like(self.prev_yaw), self.prev_yaw)
            self.yaw_sum = torch.where(reset_mask, torch.zeros_like(self.yaw_sum), self.yaw_sum)
            self.phase = torch.where(reset_mask, torch.zeros_like(self.phase), self.phase)
            self.initialized = torch.where(reset_mask, torch.zeros_like(self.initialized), self.initialized)
            self.recover_steps = torch.where(reset_mask, torch.zeros_like(self.recover_steps), self.recover_steps)
            self.cooldown = torch.where(reset_mask, torch.zeros_like(self.cooldown), self.cooldown)

    def _straight_gate(self):
        batch_idx = self.env._batch_idx
        closest_idx = self.env.closest_idx
        local_curvature = torch.abs(self.env.track_curvature[batch_idx, closest_idx])

        on_straight = local_curvature < self.curvature_threshold
        aligned = torch.abs(self.env.heading_diff) < 0.45

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

        # -------------------------
        # Faza 0: normalna jazda + przygotowanie na prostej
        # -------------------------
        drive_reward = (
                1.00 * progress
                - 0.40 * torch.abs(self.env.signed_dist_to_centerline)
                - 0.40 * torch.abs(self.env.heading_diff)
                - 0.03 * torch.abs(delta)
        )

        # Na prostych opłaca się mieć prędkość, żeby mieć z czego zrobić obrót.
        straight_speed_bonus = torch.where(
            straight,
            0.25 * torch.clamp(speed - 3.0, 0.0, 5.0),
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

        # -------------------------
        # Faza 1: obrót 360
        # -------------------------
        in_spin = self.phase == self.SPIN

        # SUMUJEMY RAW dyaw, aby uniknąć oszukiwania wibrowaniem lewo-prawo
        self.yaw_sum = torch.where(in_spin, self.yaw_sum + dyaw, self.yaw_sum)

        spin_progress = abs_dyaw / self.env.dt

        # Zmniejszona nagroda za kroki w trakcie driftu, brak nagrody za prędkość
        spin_reward = (
                0.50 * progress
                + 0.50 * torch.clamp(spin_progress, 0.0, 8.0)
                + 0.80 * torch.clamp(abs_beta, 0.0, 1.2)
                - 0.20 * torch.abs(self.env.signed_dist_to_centerline)
        )

        # Ukończenie sprawdza wartość bezwzględną z sumy kierunkowej (wymusza pełny obrót w jedną stronę)
        finished_spin = in_spin & (torch.abs(self.yaw_sum) >= self.yaw_target)

        # Ogromna nagroda za pomyślny, pełny obrót
        spin_reward = torch.where(
            finished_spin,
            spin_reward + 150.0,
            spin_reward,
        )

        self.phase = torch.where(
            finished_spin,
            torch.ones_like(self.phase) * self.RECOVER,
            self.phase,
        )

        self.recover_steps = torch.where(
            finished_spin,
            torch.zeros_like(self.recover_steps),
            self.recover_steps,
        )

        # Przerwanie obrotu, jeśli auto prawie się zatrzyma (zapobiega bączkom w miejscu)
        failed_spin = in_spin & (speed < 1.0)

        self.phase = torch.where(
            failed_spin,
            torch.zeros_like(self.phase),  # Wróć do fazy normalnej jazdy
            self.phase,
        )

        # Kara w postaci surowego cooldownu w przypadku nieudanego obrotu
        self.cooldown = torch.where(
            failed_spin,
            torch.ones_like(self.cooldown) * 200,
            self.cooldown
        )

        # Kara za mocne przekręcenie - zabezpieczona wartością bezwzględną
        overspin = torch.nn.functional.relu(torch.abs(self.yaw_sum) - 2.45 * math.pi)
        spin_reward = spin_reward - 2.0 * overspin

        # -------------------------
        # Faza 2: odzyskanie kontroli
        # -------------------------
        in_recover = self.phase == self.RECOVER
        self.recover_steps = torch.where(
            in_recover,
            self.recover_steps + 1,
            self.recover_steps,
        )

        recover_reward = (
                1.20 * progress
                - 1.00 * torch.abs(self.env.heading_diff)
                - 0.60 * abs_r
                - 0.40 * abs_beta
                - 0.30 * torch.abs(self.env.signed_dist_to_centerline)
        )

        stable_after_spin = (
                in_recover
                & (self.recover_steps > 20)
                & (torch.abs(self.env.heading_diff) < 0.30)
                & (abs_r < 0.8)
                & (speed > 2.5)
        )

        recover_reward = torch.where(
            stable_after_spin,
            recover_reward + 20.0,
            recover_reward,
        )

        self.phase = torch.where(
            stable_after_spin,
            torch.zeros_like(self.phase),
            self.phase,
        )

        # Znacznie dłuższy czas oczekiwania po udanym obrocie
        self.cooldown = torch.where(
            stable_after_spin,
            torch.ones_like(self.cooldown) * 300,
            self.cooldown,
        )

        self.yaw_sum = torch.where(
            stable_after_spin,
            torch.zeros_like(self.yaw_sum),
            self.yaw_sum,
        )

        # -------------------------
        # Wybór rewardu według fazy
        # -------------------------
        reward = torch.where(in_spin, spin_reward, drive_reward)
        reward = torch.where(in_recover, recover_reward, reward)

        # kara za lekkie wyjście poza tor
        slightly_off_track_penalty = (
                -4.0 * torch.nn.functional.relu(-self.env.signed_dist_to_edge + 0.05) ** 2.0
        )

        reward = torch.where(
            self.env.slightly_off_track,
            reward + slightly_off_track_penalty,
            reward,
        )

        reward = torch.where(
            self.env.off_track,
            torch.ones_like(reward) * -25.0,
            reward,
        )

        return reward / 10.0