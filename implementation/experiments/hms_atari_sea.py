from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.atari_env import AtariEnv
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_conv import IndividualConv
from typing import Tuple
from numpy.random import Generator, SeedSequence
import numpy as np
from ..genotype.genotype_var_len import shared_noise_table

LENGTH = 1687218


class ExperimentAtari(BaseExperiment):

    def __init__(self, encoding: str = 'var'):
        super().__init__(encoding)

    def create_individual(self, level_config: LevelConfig, rng: Generator, noise=None) -> BaseIndividual:
        gen = self.genotype(level_config.mut_prob, level_config.mut_std, LENGTH, rng, noise)
        return IndividualConv(gen, [(8, 8, 1, 32), (32,), (4, 4, 32, 64), (64,), (3, 3, 64, 64),
                                    (64,), (3136, 512), (512,), (512, 18), (18,)])

    def evaluate_individual(self, deme_id, individual_id, individual: BaseIndividual, seed: SeedSequence) -> Tuple[
        int, int, float]:
        env = AtariEnv(env_name="Frostbite-v0", seed=seed)
        return env.run_evaluation(individual, deme_id, individual_id)


def run_experiment():
    seed = 98765
    rng = np.random.default_rng(seed)

    experiment = ExperimentAtari()

    shared_noise = shared_noise_table()

    config_list = [LevelConfig(0.9, 0.005, 1000, 25, None, None)]

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', 150), n_jobs=25, rng=rng, noise=shared_noise)

    hms.run()


if __name__ == '__main__':
    run_experiment()
