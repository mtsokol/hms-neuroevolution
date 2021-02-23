from abc import ABC, abstractmethod
import os
import tensorflow as tf
import gym
from typing import List, Tuple
from ..genotype.base_individual import BaseIndividual
from numpy.random import SeedSequence
from ..environment import *

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
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.output = output
        self.no_attempts = no_attempts
        self.max_steps_per_episode = max_steps_per_episode

        tf.random.set_seed(int(10000 * self.rng.random() + 1000))

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])

    def run_episode(
            self,
            initial_state: tf.Tensor,
            model: tf.keras.Model,
            max_steps: int) -> tf.Tensor:
        """Runs a single episode to collect training data."""

        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):

            state = self.preprocess(state)

            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities
            action_logits_t = model(state)

            # Sample next action from the action probability distribution
            if self.output == 'categorical':
                action = tf.random.categorical(action_logits_t, 1)[0, 0]
            elif self.output == 'argmax':
                action = tf.math.argmax(action_logits_t, axis=1)[0]
            else:
                raise Exception('Invalid output type')

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        rewards = rewards.stack()

        return rewards

    def run_evaluation(self, individual: BaseIndividual, deme_id: int, ind_id: int) -> Tuple[int, int, float]:

        model = individual.to_phenotype()

        reward = 0.
        for i in range(self.no_attempts):
            self.env.seed(int(10000 * self.rng.random() + 1000))
            initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
            rewards = self.run_episode(initial_state, model, self.max_steps_per_episode)
            sum_reward = np.sum(rewards)
            reward += sum_reward

        result = reward / float(self.no_attempts)

        del model

        return deme_id, ind_id, result

    @abstractmethod
    def preprocess(self, obs):
        pass
