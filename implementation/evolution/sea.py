import numpy as np
from ..genotype.fixed_length import GenotypeFixedLength


class SEA:

    def __init__(self, mutation_prob: float, std: float):

        self.mutation_prob = mutation_prob
        self.std = std

    def mutate(self, genotype: GenotypeFixedLength):

        prob = np.random.rand()

        new_genotype = genotype.__copy__()
        new_genotype.gene_array = np.copy(genotype.gene_array)

        if prob < self.mutation_prob:
            new_genotype.gene_array += self.std * np.random.randn(new_genotype.length)

        return new_genotype

