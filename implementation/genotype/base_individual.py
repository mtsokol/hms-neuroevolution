from abc import ABC, abstractmethod
from .base_genotype import BaseGenotype
import numpy as np
import tensorflow as tf


class BaseIndividual(ABC):

    fitness: float

    @abstractmethod
    def __init__(self, genotype: BaseGenotype, model_shape):
        self.genotype = genotype
        self.model_shape = model_shape

    @abstractmethod
    def to_phenotype(self) -> tf.keras.Model:
        pass

    def distance_to(self, other_individual: 'BaseIndividual') -> float:
        return np.sum((self.genotype.get_gene_array() - other_individual.genotype.get_gene_array())**2)
