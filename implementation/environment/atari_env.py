import tensorflow as tf
from numpy.random import SeedSequence
from .base_env import BaseEnv
from . import grayscale_palette


class AtariEnv(BaseEnv):

    def __init__(self, env_name: str, seed: SeedSequence, no_attempts: int = 3,
                 max_steps_per_episode: int = 1000):

        super().__init__(env_name, seed, 'argmax', no_attempts, max_steps_per_episode)

    def preprocess(self, obs):
        obs = tf.gather(tf.constant(grayscale_palette), tf.cast(obs, tf.int32))
        obs = tf.reduce_max(obs, axis=2)
        resized = tf.image.resize(obs, (84, 84))
        return resized
