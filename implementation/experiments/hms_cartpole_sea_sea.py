from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.gym_env import GymEnv
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_nn import IndividualNN
from ..genotype.genotype_fixed_len import GenotypeFixedLen
from typing import Tuple
from numpy.random import Generator, SeedSequence
import numpy as np
from .hms_cartpole_sea import ExperimentCartPole

LENGTH = 58


def run_experiment():

    seed = 98765
    rng = np.random.default_rng(seed)

    experiment = ExperimentCartPole()

    config_list = [LevelConfig(0.8, 0.5, 100, 30, None, 0.5),
                   LevelConfig(0.8, 0.2, 50, 20, ('obj_no_change', 3), None)]

    hms = HMS(experiment, 2, config_list, 2, ('epochs', 8), n_jobs=10, rng=rng)

    hms.run()


if __name__ == '__main__':

    run_experiment()
