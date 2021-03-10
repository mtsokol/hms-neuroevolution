from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..visualization import DIR
import numpy as np
from .hms_cartpole_sea import ExperimentCartPole
from . import run_arg_parser, create_client, create_exit_handler


def run_experiment(seed, n_jobs, epochs):

    client = create_client(n_jobs)

    experiment = ExperimentCartPole(encoding='var')

    config_list = [LevelConfig(0.8, 0.5, 150, 30, None, None)]

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', epochs), n_jobs=n_jobs, seed=seed, out_dir=DIR)

    future = client.submit(hms.run)

    create_exit_handler(future)

    logs = future.result()

    hms.log_summary_metrics(logs)

    input("Press any key to exit...")


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
