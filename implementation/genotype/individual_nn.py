from .base_individual import BaseIndividual
from .base_genotype import BaseGenotype
import numpy as np
import tensorflow as tf


class IndividualNN(BaseIndividual):

    def __init__(self, genotype: BaseGenotype, model_shape):
        super().__init__(genotype, model_shape)

    def to_phenotype(self) -> tf.keras.Model:
        flat_shapes = list(map(lambda tup: np.prod(tup), self.model_shape))
        flat_weights = tf.split(tf.constant(self.genotype.get_gene_array()), flat_shapes)
        weights = list(map(lambda tup: tf.reshape(*tup), zip(flat_weights, self.model_shape)))

        model = tf.keras.Sequential()

        for i in range(0, len(self.model_shape), 2):

            i_1, i_2 = self.model_shape[i]

            if i == 0:
                model.add(tf.keras.layers.Dense(i_2, activation="relu", input_shape=(i_1,)))
            elif i == len(self.model_shape) - 2:
                model.add(tf.keras.layers.Dense(i_2))
            else:
                model.add(tf.keras.layers.Dense(i_2, activation="relu"))

        model.set_weights(weights)

        return model
