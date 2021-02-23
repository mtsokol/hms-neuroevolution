from numpy.random import SeedSequence
from .base_env import BaseEnv


class GymEnv(BaseEnv):

    def __init__(self, env_name: str, seed: SeedSequence, no_attempts: int = 15,
                 max_steps_per_episode: int = 1000):

        super().__init__(env_name, seed, 'categorical', no_attempts, max_steps_per_episode)

    def preprocess(self, obs):
        return obs
