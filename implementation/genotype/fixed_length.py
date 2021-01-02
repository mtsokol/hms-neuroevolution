import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math


class GenotypeFixedLength:

    def __copy__(self):
        return GenotypeFixedLength(self.length, self.net_shapes)

    def __init__(self, length: int, net_shapes: list):

        self.length = length
        self.net_shapes = net_shapes
        self.gene_array = np.random.randn(length) * 0.8
        self.last_fitness = -math.inf

    def to_phenotype(self):

        flat_shapes = list(map(lambda x: np.prod(x), self.net_shapes))
        flat_weights = tf.split(tf.constant(self.gene_array), flat_shapes)
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
