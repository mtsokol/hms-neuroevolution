from .base_genotype import BaseGenotype
from numpy.random import Generator
import numpy as np
from copy import deepcopy
from numpy.lib.format import open_memmap
import os


class GenotypeVarLen(BaseGenotype):

    def __init__(self, mut_prob: float, mut_std: float, gen_len: int, rng: Generator):
        self.mut_prob = mut_prob
        self.mut_std = mut_std
        self.rng = rng
        self.gen_len = gen_len
        self.seeds = [int(100000 * self.rng.random() + 1)]

    def mutate(self) -> None:
        if self.rng.random() < self.mut_prob:
            self.seeds.append(int(100000 * self.rng.random() + 1))

    def crossover(self, other_genotype: 'BaseGenotype') -> None:
        raise NotImplementedError()

    def get_gene_array(self) -> np.ndarray:

        noise = open_memmap(os.environ['NOISE_PATH'], mode='r', dtype=np.float32)

        built_gene_array = np.copy(noise[self.seeds[0]:self.seeds[0]+self.gen_len])

        for seed in self.seeds[1:]:
            built_gene_array += self.mut_std * noise[seed:seed+self.gen_len]

        return built_gene_array

    def __deepcopy__(self, memodict={}):
        copied_gen = GenotypeVarLen(self.mut_prob, self.mut_std, self.gen_len, self.rng)
        copied_gen.seeds = deepcopy(self.seeds)
        return copied_gen
