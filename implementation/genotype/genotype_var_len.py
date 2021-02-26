from .base_genotype import BaseGenotype
from numpy.random import Generator
import numpy as np
import multiprocessing as mp
import ctypes
from copy import deepcopy


def shared_noise_table() -> np.ndarray:
    seed = 123
    count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
    print('Sampling {} random numbers with seed {}'.format(count, seed))
    shared_mem = mp.Array(ctypes.c_float, count, lock=False)
    noise = np.ctypeslib.as_array(shared_mem)
    assert noise.dtype == np.float32
    noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
    print('Sampled {} bytes'.format(noise.size * 4))
    return noise


class GenotypeVarLen(BaseGenotype):

    def __init__(self, mut_prob: float, mut_std: float, gen_len: int, rng: Generator, noise):
        self.mut_prob = mut_prob
        self.mut_std = mut_std
        self.rng = rng
        self.gen_len = gen_len
        self.seeds = [int(100000 * self.rng.random() + 1)]  # TODO initial seed here
        self.noise = noise

    def mutate(self) -> None:
        if self.rng.random() < self.mut_prob:
            self.seeds.append(int(100000 * self.rng.random() + 1))

    def crossover(self, other_genotype: 'BaseGenotype') -> None:
        raise NotImplementedError()

    def get_gene_array(self) -> np.ndarray:

        built_gene_array = np.copy(self.noise[self.seeds[0]:self.seeds[0]+self.gen_len])

        for seed in self.seeds[1:]:
            built_gene_array += self.mut_std * self.noise[seed:seed+self.gen_len]

        return built_gene_array

    def __deepcopy__(self, memodict={}):
        copied_gen = GenotypeVarLen(self.mut_prob, self.mut_std, self.gen_len, self.rng, self.noise)
        copied_gen.seeds = deepcopy(self.seeds)
        return copied_gen
