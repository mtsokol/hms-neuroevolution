from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.gym_env import GymEnv
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_nn import IndividualNN
from typing import Tuple
from numpy.random import Generator, SeedSequence
import numpy as np

LENGTH = 58


class ExperimentCartPole(BaseExperiment):

    def __init__(self, encoding: str = 'fixed'):
        super().__init__(encoding)

    def create_individual(self, level_config: LevelConfig, rng: Generator, noise=None) -> BaseIndividual:
        gen = self.genotype(level_config.mut_prob, level_config.mut_std, LENGTH, rng, noise)
        return IndividualNN(gen, [(4, 8), (8,), (8, 2), (2,)])

    def evaluate_individual(self, deme_id, individual_id, individual: BaseIndividual, seed: SeedSequence) -> Tuple[
        int, int, float]:
        env = GymEnv(env_name="CartPole-v0", seed=seed)
        return env.run_evaluation(individual, deme_id, individual_id)


def run_experiment():
    seed = 98765
    rng = np.random.default_rng(seed)

    experiment = ExperimentCartPole()

    config_list = [LevelConfig(0.8, 0.5, 150, 30, None, None)]

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', 10), n_jobs=8, rng=rng)

    hms.run()


if __name__ == '__main__':
    run_experiment()
