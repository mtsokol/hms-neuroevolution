from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..visualization.utils import save_experiment_description
from numpy.random import Generator
import numpy as np
from .hms_cartpole_sea import ExperimentCartPole
from . import run_arg_parser


def run_experiment(seed, n_jobs, epochs):

    rng = np.random.default_rng(seed)

    experiment = ExperimentCartPole()

    config_list = [LevelConfig(0.8, 0.5, 110, 30, None, 0.5),
                   LevelConfig(0.8, 0.2, 40, 20, ('obj_no_change', 3), None)]

    hms = HMS(experiment, 2, config_list, 2, ('epochs', epochs), n_jobs=n_jobs, rng=rng)

    save_experiment_description(hms, type(experiment).__name__, seed)

    hms.run()


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
