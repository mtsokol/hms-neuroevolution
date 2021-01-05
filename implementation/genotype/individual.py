from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple
import math


class Individual(object):

    def __init__(self, length: int, net_shapes: List[Tuple], level: int):
        self.length: int = length
        self.net_shapes: List[Tuple] = net_shapes
        self.level: int = level
        self.genotype_array: np.ndarray = np.random.randn(length) * 0.8
        self.last_fitness: float = -math.inf

    def to_phenotype(self) -> tf.keras.Model:

        flat_shapes = list(map(lambda x: np.prod(x), self.net_shapes))
        flat_weights = tf.split(tf.constant(self.genotype_array), flat_shapes)
        weights = list(map(lambda tup: tf.reshape(*tup), zip(flat_weights, self.net_shapes)))

        model = tf.keras.Sequential()

        for i in range(0, len(self.net_shapes), 2):

            i_1, i_2 = self.net_shapes[i]

            if i == 0:
                model.add(layers.Dense(i_2, activation="relu", input_shape=(i_1,)))
            elif i == len(self.net_shapes)-2:
                model.add(layers.Dense(i_2))
            else:
                model.add(layers.Dense(i_2, activation="relu"))

        model.set_weights(weights)

        return model

    def distance_to(self, other_genotype: Individual):
        return np.sum(np.abs(self.genotype_array - other_genotype.genotype_array))

    def increment_level(self):
        self.level += 1
