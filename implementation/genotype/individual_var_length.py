import numpy as np


class IndividualVarLength:

    def __init__(self):
        pass

    def compute_mutation(self, noise, parent_theta, idx, mutation_power):
        return parent_theta + mutation_power * noise.get(idx, self.num_params)

    def mutate(self, parent, rs, noise, mutation_power):
        parent_theta, parent_seeds = parent
        idx = noise.sample_index(rs, self.num_params)
        seeds = parent_seeds + ((idx, mutation_power), )
        theta = self.compute_mutation(noise, parent_theta, idx, mutation_power)
        return theta, seeds
