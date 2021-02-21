from .deme import Deme
from typing import Tuple
from ...experiments.base_experiment import BaseExperiment
from collections import OrderedDict
from joblib import Parallel, delayed
from uuid import uuid1, UUID
from copy import deepcopy
from numpy.random import SeedSequence
from ...visualization.plotting import *
from .config import LevelConfig


class HMS:

    def __init__(self,
                 experiment: BaseExperiment,
                 levels: int,
                 config_list: List[LevelConfig],
                 metaepoch_length: int,
                 gsc: Tuple[str, int],
                 n_jobs,
                 rng):
        assert levels == len(config_list), 'dims don\'t match'

        self.experiment = experiment
        self.levels = levels
        self.config_list = config_list
        self.demes: OrderedDict[UUID, Deme] = OrderedDict()
        self.gsc = gsc
        self.epoch = 1
        self.metaepoch = 1
        self.metaepoch_length = metaepoch_length
        self.n_jobs = n_jobs
        self.rng = rng
        self.executor = Parallel(n_jobs=self.n_jobs)
        self.seed_seq = SeedSequence(int(10000 * rng.random() + 1000))

    def run(self):

        self.__initialize_root_deme()

        while not self.gsc_satisfied():

            print(f'Starting epoch {self.epoch}')

            if self.epoch % self.metaepoch_length == 0:
                print(f'Starting metaepoch in {self.epoch}')
                self.__perform_sprouting()

            self.__evaluate_individuals()

            self.__log_metrics()

            self.__run_step()

            self.epoch += 1

    def __initialize_root_deme(self):

        root_deme = self.__create_deme(level=0)
        self.demes[uuid1()] = root_deme

    def __evaluate_individuals(self):

        jobs_to_evaluate = []

        for deme_id, deme in self.demes.items():
            jobs = deme.get_jobs()
            seeds = self.seed_seq.spawn(len(jobs))

            for (ind_id, individual), seed in zip(jobs, seeds):
                jobs_to_evaluate.append((deme_id, ind_id, individual, seed))

        results = self.executor(delayed(self.experiment.evaluate_individual)(*job) for job in jobs_to_evaluate)

        for deme_id, ind_id, fitness in results:
            self.demes[deme_id].set_fitness(ind_id, fitness)

    def __run_step(self):

        for deme_id, deme in self.demes.items():
            deme.run_step()

    def __create_deme(self, level: int, initial_individuals: list = None) -> Deme:

        config = self.config_list[level]

        if initial_individuals is None:
            pop = [self.experiment.create_individual(config, self.rng) for _ in range(config.pop_size)]
        else:
            pop = []
            for _ in range(config.pop_size):
                ind = self.rng.choice(initial_individuals)
                ind = deepcopy(ind)
                ind.genotype.mutate()
                pop.append(ind)

        return Deme(level, pop, config, self.rng)

    def __perform_sprouting(self) -> None:

        for deme_id, deme in self.demes.items():

            if deme.level == self.levels + 1:
                continue

            min_dist = self.config_list[deme.level].spr_cond

            elites = []
            for other_deme_id, other_deme in self.demes.items():
                if not other_deme_id == deme_id and other_deme.elite is not None:
                    elites.append(other_deme.elite)

            can_sprout = True
            for other_elite in elites:
                if other_elite.distance_to(deme.elite) <= min_dist:
                    can_sprout = False

            if can_sprout:
                new_deme = self.__create_deme(deme.level + 1, [deme.elite])
                self.demes[uuid1()] = new_deme

    def gsc_satisfied(self) -> bool:
        for _, deme in self.demes.items():
            if deme.alive:
                return False
        return True

    def __log_metrics(self):

        for _, deme in self.demes.items():
            if deme.alive:
                elite_score = max(deme.population.values(), key=lambda ind: ind.fitness).fitness
                scores = list(map(lambda ind: ind.fitness, deme.population.values()))

                plot_histogram_with_elite(scores, elite_score, self.epoch)
