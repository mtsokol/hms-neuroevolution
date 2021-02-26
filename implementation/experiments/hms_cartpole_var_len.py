from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..genotype.genotype_var_len import shared_noise_table
from ..visualization.utils import save_experiment_description
from numpy.random import Generator
import numpy as np
from .hms_cartpole_sea import ExperimentCartPole
from . import run_arg_parser


def run_experiment(seed, n_jobs, epochs):

    rng = np.random.default_rng(seed)

    shared_noise = shared_noise_table()

    experiment = ExperimentCartPole(encoding='var')

    config_list = [LevelConfig(0.8, 0.5, 150, 30, None, None)]

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', epochs), n_jobs=n_jobs, rng=rng, noise=shared_noise)

    save_experiment_description(hms, type(experiment).__name__, seed)

    hms.run()


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
