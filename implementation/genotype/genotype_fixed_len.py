from .base_genotype import BaseGenotype
from numpy.random import Generator
import numpy as np
from copy import deepcopy


class GenotypeFixedLen(BaseGenotype):

    def __init__(self, mut_prob: float, mut_std: float, gen_len: int, rng: Generator):
        self.mut_prob = mut_prob
        self.mut_std = mut_std
        self.rng = rng
        self.gen_len = gen_len
        self.genotype_array: np.ndarray = self.rng.standard_normal(gen_len)

    def mutate(self) -> None:
        if self.rng.random() < self.mut_prob:
            self.genotype_array += self.mut_std * self.rng.standard_normal(self.gen_len)

    def crossover(self, other_genotype: 'BaseGenotype') -> None:
        raise NotImplementedError()

    def get_gene_array(self) -> np.ndarray:
        return self.genotype_array

    def __deepcopy__(self, memodict={}):
        copied_gen = GenotypeFixedLen(self.mut_prob, self.mut_std, self.gen_len, self.rng)
        copied_gen.genotype_array = deepcopy(self.genotype_array)
        return copied_gen
