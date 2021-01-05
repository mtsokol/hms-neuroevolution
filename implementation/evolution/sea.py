import numpy as np
from ..genotype.individual import Individual
from copy import deepcopy


class SEA:

    def __init__(self, mutation_prob: float, std: float):

        self.mutation_prob = mutation_prob
        self.std = std

    def mutate(self, genotype: Individual):

        prob = np.random.rand()

        new_genotype = deepcopy(genotype)

        if prob < self.mutation_prob:
            new_genotype.genotype_array += self.std * np.random.randn(new_genotype.length)

        return new_genotype
