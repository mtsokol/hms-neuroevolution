from .base_individual import BaseIndividual
from .base_genotype import BaseGenotype
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class IndividualConv(BaseIndividual):

    def __init__(self, genotype: BaseGenotype, model_shape):
        super().__init__(genotype, model_shape)

    def to_phenotype(self) -> tf.keras.Model:
        flat_shapes = list(map(lambda tup: np.prod(tup), self.model_shape))
        flat_weights = tf.split(tf.constant(self.genotype.get_gene_array()), flat_shapes)
        weights = list(map(lambda tup: tf.reshape(*tup), zip(flat_weights, self.model_shape)))

        model = tf.keras.Sequential()

        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.model_shape[-1][0]))

        model.set_weights(weights)

        return model
