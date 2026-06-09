from racing_env.envs.gymnasium import SingleTrackVecEnv

class SingleTrackVecTorchEnv(SingleTrackVecEnv):
    def step(self, actions):
        state, reward, observation, terminated, truncated, info = self.simulator.forward(actions)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        state, info, observation = self.simulator.reset()

        if self.screen_renderer._rendering_started:
            self.screen_renderer.set_friction_map(self.simulator.get_friction_map())

        return observation, info