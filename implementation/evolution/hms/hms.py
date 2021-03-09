from .deme import Deme
from typing import Tuple, Optional, List
from ...experiments.base_experiment import BaseExperiment
from ...genotype.base_individual import BaseIndividual
from collections import OrderedDict
from dask.distributed import get_client
from uuid import uuid1, UUID
from copy import deepcopy
from numpy.random import SeedSequence
from ...visualization import utils, plotting
from .config import LevelConfig
import numpy as np
from ...visualization.utils import save_experiment_description


class HMS:

    def __init__(self,
                 experiment: BaseExperiment,
                 levels: int,
                 config_list: List[LevelConfig],
                 metaepoch_length: int,
                 gsc: Tuple[str, int],
                 n_jobs: int,
                 seed,
                 noise=None):
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
        self.rng = np.random.default_rng(seed)
        self.seed_seq = SeedSequence(int(10000 * self.rng.random() + 1000))
        self.noise = noise
        self.elite_score_history = []

        save_experiment_description(self, type(experiment).__name__, seed)

    def run(self):

        self.__initialize_root_deme()

        while not self.gsc_satisfied():

            print(f'Starting epoch {self.epoch}')

            if self.epoch % self.metaepoch_length == 0:
                print(f'Starting metaepoch in {self.epoch}')
                self.__perform_sprouting()

            self.__evaluate_individuals()

            self.__evaluate_hms_elite()

            # self.__log_epoch_metrics()

            self.__run_step()

            self.epoch += 1

        return self.elite_score_history

    def __initialize_root_deme(self):

        root_deme = self.__create_deme(level=0)
        self.demes[uuid1()] = root_deme

    def __evaluate_individuals(self):

        jobs_list = [[] for _ in range(4)]

        for deme_id, deme in self.demes.items():
            jobs = deme.get_jobs()
            seeds = self.seed_seq.spawn(len(jobs))

            for (ind_id, individual), seed in zip(jobs, seeds):
                jobs_list[0].append(deme_id)
                jobs_list[1].append(ind_id)
                jobs_list[2].append(individual)
                jobs_list[3].append(seed)

        client = get_client()
        futures = client.map(self.experiment.evaluate_individual, *jobs_list)
        results = client.gather(futures)

        for deme_id, ind_id, fitness in results:
            self.demes[deme_id].set_fitness(ind_id, fitness)

        for deme in self.demes.values():
            deme.update_elite()

    def __run_step(self):

        for deme_id, deme in self.demes.items():
            deme.run_step()

    def __create_deme(self, level: int, initial_individuals: list = None) -> Deme:

        config = self.config_list[level]

        if initial_individuals is None:
            pop = [self.experiment.create_individual(config, self.rng, self.noise) for _ in range(config.pop_size)]
        else:
            pop = []
            for _ in range(config.pop_size):
                ind = self.rng.choice(initial_individuals)
                ind = deepcopy(ind)
                ind.genotype.mutate()
                pop.append(ind)

        return Deme(level, pop, config, self.rng)

    def __perform_sprouting(self) -> None:

        new_demes = []

        for deme_id, deme in self.demes.items():

            min_dist = self.config_list[deme.level].spr_cond

            if deme.level == self.levels - 1 or min_dist is None or not deme.alive:
                continue

            elites = []
            for other_deme_id, other_deme in self.demes.items():
                if other_deme.alive and not other_deme_id == deme_id and other_deme.elite is not None:
                    elites.append(other_deme.elite)

            can_sprout = True
            for other_elite in elites:
                if other_elite.distance_to(deme.elite) <= min_dist:
                    can_sprout = False

            if can_sprout:
                new_deme = self.__create_deme(deme.level + 1, [deme.elite])
                new_demes.append(new_deme)

        for new_deme in new_demes:
            self.demes[uuid1()] = new_deme

    def gsc_satisfied(self) -> bool:

        gsc_type, value = self.gsc

        if gsc_type == 'deme_alive':
            for _, deme in self.demes.items():
                if deme.alive:
                    return False
            return True
        elif gsc_type == 'epochs':
            return self.epoch >= value
        else:
            raise Exception('Invalid GSC')

    def __log_epoch_metrics(self):  # TODO proper metrics

        for deme_id, deme in self.demes.items():
            if deme.alive:
                scores = list(map(lambda ind: ind.fitness, deme.population.values()))
                plotting.plot_histogram_with_elite(scores, deme.elite.fitness, self.epoch, deme_id)

    def log_summary_metrics(self, elite_score_history):

        if len(elite_score_history) > 0:
            plotting.plot_median_with_intervals(elite_score_history, self.rng)

    def __evaluate_hms_elite(self):

        hms_elite: Optional[BaseIndividual] = None

        for deme in self.demes.values():
            if deme.alive:
                if hms_elite is None or hms_elite.fitness < deme.elite.fitness:
                    hms_elite = deme.elite

        jobs_list = [[] for _ in range(4)]
        seeds = self.seed_seq.spawn(5)

        for seed in seeds:
            jobs_list[0].append(None)
            jobs_list[1].append(None)
            jobs_list[2].append(hms_elite)
            jobs_list[3].append(seed)

        client = get_client()
        futures = client.map(self.experiment.evaluate_individual, *jobs_list)
        results = client.gather(futures)
        results = list(map(lambda x: x[2], results))
        self.elite_score_history.append(results)

        # save elite model
        utils.save_model(hms_elite, self.epoch)

    def __str__(self):

        specs = f'''HMS config:
        \tlevels: {self.levels}
        \tmetaepoch_length: {self.metaepoch_length}
        \tgsc: {self.gsc}
        \tn_jobs: {self.n_jobs}\n'''

        for config in self.config_list:
            specs += str(config)

        return specs
