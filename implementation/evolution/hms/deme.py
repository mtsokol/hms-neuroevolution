import numpy as np
from typing import Optional
from ...genotype.individual import Individual


class Deme:

    def __init__(self, level: int, population: list, executor, evaluate_individual, alg):
        self.level = level
        self.population = population
        self.elite: Optional[Individual] = None
        self.executor = executor
        self.evaluate_individual = evaluate_individual
        self.alg = alg
        self.futures = None
        self.history = dict()

    def run_evaluations(self):
        self.futures = [self.executor.submit(self.evaluate_individual, individual) for individual in self.population]
        return self.futures

    def collect_evaluations(self, epoch: int, promoted_num: int):
        pop = list(map(lambda d: d.result(), self.futures))
        pop_sorted = sorted(pop, key=lambda ind: -ind.last_fitness)
        promoted = pop_sorted[:promoted_num]
        pop = [self.alg.mutate(promoted[i]) for i in np.random.randint(low=0, high=promoted_num, size=len(self.population)-1)]
        pop += promoted[:1]
        self.population = pop
        self.futures = None
        self.elite = promoted[0]

        self.history[epoch] = self.elite

        print(f"best fitness for deme {hash(self)} in epoch {epoch} is {self.elite.last_fitness}")
