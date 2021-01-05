import numpy as np
import concurrent
from .deme import Deme


class HMS:

    def __init__(self,
                 levels: int,
                 create_individual,
                 evaluate_individual,
                 alg_list: list,
                 pop_sizes: list,
                 lsc_list: list,
                 spr_list: list,
                 gsc_val: int,
                 metaepoch_length: int,
                 executor):
        assert levels == len(alg_list) == len(pop_sizes) == len(lsc_list) == len(spr_list), 'dims don\'t match'

        self.levels = levels
        self.create_individual = create_individual
        self.evaluate_individual = evaluate_individual
        self.pop_sizes = pop_sizes
        self.alg_list = alg_list
        self.spr_list = spr_list
        self.running_demes = []
        self.gsc_val = gsc_val
        self.epoch = 1
        self.metaepoch = 1
        self.metaepoch_length = metaepoch_length
        self.executor = executor
        self.best_in_epoch = None

    def run(self):

        root_deme = self.create_deme(level=0)
        self.running_demes.append(root_deme)

        while not self.gsc_satisfied():

            print(f'Starting epoch {self.epoch}')

            if self.epoch % self.metaepoch_length == 0:
                print(f'Starting metaepoch in {self.epoch}')
                for deme in self.running_demes:
                    if self.can_sprout(deme):
                        print('sprouting')
                        self.perform_sprout(deme)

            demes_futures = [deme.run_evaluations() for deme in self.running_demes]
            future_to_wait = [f for futures in demes_futures for f in futures]
            concurrent.futures.wait(future_to_wait)
            [deme.collect_evaluations(self.epoch, 30) for deme in self.running_demes]

            self.epoch += 1

    def create_deme(self, level: int, initial_individuals: list = None) -> Deme:
        alg = self.alg_list[level]

        if initial_individuals is None:
            pop = [self.create_individual(level) for _ in range(self.pop_sizes[level])]
        else:
            init_size = len(initial_individuals)
            pop = [alg.mutate(initial_individuals[i]) for i in
                   np.random.randint(low=0, high=init_size, size=self.pop_sizes[level] - 1)]
            pop += initial_individuals[:1]

            list(map(lambda i: i.increment_level(), pop))

        return Deme(level, pop, self.executor, self.evaluate_individual, alg)

    def can_sprout(self, deme_to_sprout) -> bool:

        if deme_to_sprout.level == self.levels-1:
            return False

        demes = list(self.running_demes)
        demes.remove(deme_to_sprout)

        for deme in demes:
            if deme.level == deme_to_sprout.level+1 and deme.elite.distance_to(deme_to_sprout.elite) < self.spr_list[deme.level]:
                return False

        return True

    def perform_sprout(self, deme):
        new_deme = self.create_deme(deme.level+1, [deme.elite])
        self.running_demes.append(new_deme)

    def gsc_satisfied(self) -> bool:
        for deme in self.running_demes:
            if deme.elite is not None and deme.elite.last_fitness >= self.gsc_val:
                return True
        return False
