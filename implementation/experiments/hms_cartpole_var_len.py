from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.gym_env import GymEnv
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_nn import IndividualNN
from ..genotype.genotype_var_len import shared_noise_table
from typing import Tuple
from numpy.random import Generator, SeedSequence
import numpy as np
from .hms_cartpole_sea import ExperimentCartPole

LENGTH = 58


def run_experiment():

    seed = 9876
    rng = np.random.default_rng(seed)

    shared_noise = shared_noise_table()

    experiment = ExperimentCartPole(encoding='var')

    config_list = [LevelConfig(0.8, 0.5, 150, 30, None, None)]

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', 6), n_jobs=10, rng=rng, noise=shared_noise)

    hms.run()


if __name__ == '__main__':

    run_experiment()
