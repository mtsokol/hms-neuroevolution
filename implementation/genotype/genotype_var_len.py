from .base_genotype import BaseGenotype
from numpy.random import Generator
import numpy as np
import multiprocessing as mp
import ctypes


class SharedNoiseTable(object):

    def __init__(self):
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        print('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = mp.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        print('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def __deepcopy__(self, memodict={}):
        print('nice try :)')
        return self


class GenotypeVarLen(BaseGenotype):

    def __init__(self, mut_prob: float, mut_std: float, gen_len: int, rng: Generator, noise: SharedNoiseTable):
        self.mut_prob = mut_prob
        self.mut_std = mut_std
        self.rng = rng
        self.gen_len = gen_len
        self.seeds = []  # TODO initial seed here
        self.noise = noise

    def mutate(self) -> None:
        # TODO if above mut_prob add next seed
        pass

    def crossover(self, other_genotype: 'BaseGenotype') -> None:
        raise NotImplementedError()

    def get_gene_array(self) -> np.ndarray:

        built_gene_array = self.noise.get(self.seeds[0], self.gen_len)

        for seed in self.seeds[1:]:
            built_gene_array += self.mut_std * self.noise.get(seed, self.gen_len)

        return built_gene_array
