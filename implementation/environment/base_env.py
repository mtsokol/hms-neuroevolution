from abc import ABC, abstractmethod
import os
import numpy as np
import tensorflow as tf
import gym
from typing import Tuple
from ..genotype.base_individual import BaseIndividual
from numpy.random import SeedSequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class BaseEnv(ABC):

    def __init__(self,
                 env_name: str,
                 seed: SeedSequence,
                 output: str,  # 'argmax' or 'categorical'
                 no_attempts: int = 15,
                 max_steps_per_episode: int = 1000
                 ):
        env = gym.make(env_name)
        self.num_actions = env.action_space.n
        self.num_obs_space = env.observation_space.shape[0]
        self.envs = []
        self.obses = []
        self.dones = np.full(no_attempts, False)
        self.rng = np.random.default_rng(seed)
        self.output = output
        self.no_attempts = no_attempts
        self.max_steps_per_episode = max_steps_per_episode

        for i in range(no_attempts):
            env = gym.make(env_name)
            env.seed(int(10_000_000 * self.rng.random() + 1000))
            obs = env.reset()
            self.envs.append(env)
            self.obses.append(obs)

        tf.random.set_seed(int(10_000_000 * self.rng.random() + 1000))

    def run_episode(
            self,
            model: tf.keras.Model,
            max_steps: int) -> float:

        reward_sum = 0.
        count = 0

        while not np.all(self.dones) and count < max_steps:

            indices = np.where(~self.dones)[0]
            state = np.array([self.preprocess(self.obses[i]) for i in indices])

            action_logits = model(state)

            if self.output == 'categorical':
                actions = tf.squeeze(tf.random.categorical(action_logits, 1), axis=1)
            elif self.output == 'argmax':
                actions = tf.math.argmax(action_logits, axis=1)
            else:
                raise Exception('Invalid output type')

            for idx, action in zip(indices, actions):
                obs, reward, done, _ = self.envs[idx].step(action.numpy())
                self.obses[idx] = obs
                self.dones[idx] = done
                reward_sum += reward

            count += 1

        return reward_sum / float(self.no_attempts)

    def run_evaluation(self, individual: BaseIndividual, deme_id: int, ind_id: int) -> Tuple[int, int, float]:
        model = individual.to_phenotype()
        reward = self.run_episode(model, self.max_steps_per_episode)
        del model
        return deme_id, ind_id, reward

    @abstractmethod
    def preprocess(self, obs):
        pass
