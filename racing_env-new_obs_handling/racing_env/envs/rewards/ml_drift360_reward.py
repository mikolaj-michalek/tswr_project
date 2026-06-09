import math
import torch


class Drift360Reward:
    def __init__(
        self,
        env,
        spin_coeff=2.0,
        drift_coeff=0.5,
        progress_coeff=0.2,
        after_360_progress_coeff=2.0,
        yaw_target=2.0 * math.pi,
        complete_bonus=20.0,
        off_track_coeff=-5.0,
        overspin_coeff=-2.0,
    ):
        self.env = env

        self.spin_coeff = spin_coeff
        self.drift_coeff = drift_coeff
        self.progress_coeff = progress_coeff
        self.after_360_progress_coeff = after_360_progress_coeff
        self.yaw_target = yaw_target
        self.complete_bonus = complete_bonus
        self.off_track_coeff = off_track_coeff
        self.overspin_coeff = overspin_coeff

        self.prev_yaw = torch.zeros(self.env.num_envs, device=self.env.device)
        self.yaw_sum = torch.zeros(self.env.num_envs, device=self.env.device)
        self.finished_360 = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        self.initialized = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )

    def _reset_finished_envs(self):
        """
        Resetuje wewnętrzne liczniki rewardu po resecie epizodu.
        Zakładamy, że env ma licznik steps.
        """
        if hasattr(self.env, "steps"):
            reset_mask = self.env.steps <= 1

            self.yaw_sum = torch.where(
                reset_mask,
                torch.zeros_like(self.yaw_sum),
                self.yaw_sum,
            )

            self.finished_360 = torch.where(
                reset_mask,
                torch.zeros_like(self.finished_360),
                self.finished_360,
            )

            self.initialized = torch.where(
                reset_mask,
                torch.zeros_like(self.initialized),
                self.initialized,
            )

    def __call__(self):
        self._reset_finished_envs()

        state = self.env.state

        yaw = state[:, 2]
        v_x = state[:, 3]
        v_y = state[:, 4]
        r = state[:, 5]

        # Pierwsze wywołanie po resecie: tylko zapamiętujemy yaw,
        # żeby nie nabić sztucznego dyaw.
        self.prev_yaw = torch.where(self.initialized, self.prev_yaw, yaw)
        self.initialized = torch.ones_like(self.initialized)

        dyaw = yaw - self.prev_yaw
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        self.prev_yaw = yaw.clone()

        abs_dyaw = torch.abs(dyaw)
        self.yaw_sum = self.yaw_sum + abs_dyaw

        # Kąt poślizgu beta: duży beta = auto jedzie bokiem.
        beta = torch.atan2(v_y, torch.clamp(torch.abs(v_x), min=0.1))
        abs_beta = torch.abs(beta)

        progress_reward = self.env.progress / self.env.dt

        # Przed wykonaniem 360:
        # - nagradzamy narastanie obrotu,
        # - nagradzamy poślizg,
        # - dajemy małą nagrodę za progres, żeby nie stał w miejscu.
        before_360_reward = (
            self.spin_coeff * abs_dyaw / self.env.dt
            + self.drift_coeff * abs_beta
            + self.progress_coeff * progress_reward
        )

        just_finished = (self.yaw_sum >= self.yaw_target) & (~self.finished_360)

        # Po 360:
        # - nagradzamy jazdę dalej,
        # - karzemy dalsze kręcenie,
        # - karzemy dalszy duży poślizg.
        after_360_reward = (
            self.after_360_progress_coeff * progress_reward
            - 0.5 * torch.abs(r)
            - 0.2 * abs_beta
        )

        reward = torch.where(
            self.finished_360,
            after_360_reward,
            before_360_reward,
        )

        reward = torch.where(
            just_finished,
            reward + self.complete_bonus,
            reward,
        )

        self.finished_360 = self.finished_360 | just_finished

        # Kara za próbę robienia 720 albo kręcenia się bez końca.
        overspin = torch.nn.functional.relu(self.yaw_sum - 2.3 * math.pi)
        reward = reward + self.overspin_coeff * overspin

        # Kara za bliskość / przekroczenie krawędzi, analogicznie do basic_reward.
        slightly_off_track_penalty = (
            self.off_track_coeff
            * torch.nn.functional.relu(-self.env.signed_dist_to_edge + 0.05) ** 2.0
        )

        reward = torch.where(
            self.env.slightly_off_track,
            reward + slightly_off_track_penalty,
            reward,
        )

        reward = torch.where(
            self.env.off_track,
            torch.ones_like(reward) * self.off_track_coeff,
            reward,
        )

        return reward / 10.0