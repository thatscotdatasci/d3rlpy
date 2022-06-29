import gym
import numpy as np


class HalfCheetahObsNoise(gym.ObservationWrapper):
    def __init__(self, env, noise_std: float = 0.):
        super().__init__(env)
        self.noise_std = float(noise_std)

    def observation(self, obs):
        noise = np.concatenate(
                [
                    self.env.np_random.randn(self.model.nq)[1:],
                    self.env.np_random.randn(self.model.nv),
                ]
            )
        noise *= self.noise_std
        return obs + noise


class HalfCheetahPO(gym.ObservationWrapper):
    def __init__(self, env, masked_indices: list):
        super().__init__(env)
        self.masked_indices = masked_indices

    def observation(self, obs):
        obs[self.masked_indices] = 0.
        return obs
