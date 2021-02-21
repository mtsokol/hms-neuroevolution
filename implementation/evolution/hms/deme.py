from typing import Optional
from ...genotype.base_individual import BaseIndividual
from .config import LevelConfig
from collections import OrderedDict
from uuid import uuid1, UUID
import numpy as np
from copy import deepcopy


class Deme:

    def __init__(self, level: int, initial_population: list, level_config: LevelConfig, rng):
        self.level = level
        self.population: OrderedDict[UUID, BaseIndividual] = OrderedDict()
        self.alive = True
        self.elite: Optional[BaseIndividual] = None
        self.level_config = level_config
        self.rng = rng

        for ind in initial_population:
            self.population[uuid1()] = ind

    def set_fitness(self, ind_id: UUID, fitness: np.float):
        self.population[ind_id].fitness = fitness

    def get_jobs(self):
        if self.alive:
            return self.population.items()
        else:
            return []

    def run_step(self):
        individuals = list(self.population.values())
        individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        self.elite = individuals[0]

        new_population = OrderedDict()

        for _ in range(self.level_config.pop_size-1):
            ind = self.rng.choice(individuals[:self.level_config.promoted_num])
            ind = deepcopy(ind)
            ind.genotype.mutate()
            new_population[uuid1()] = ind

        new_population[uuid1()] = self.elite

        del self.population
        self.population = new_population

        print(f"best fitness for deme {hash(self)} is {self.elite.fitness}")
