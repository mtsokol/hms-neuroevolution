from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..visualization import DIR
from .hms_cartpole_sea import ExperimentCartPole
from . import run_arg_parser, create_client, create_exit_handler


def run_experiment(seed, n_jobs, epochs):

    client = create_client(n_jobs)

    experiment = ExperimentCartPole()

    config_list = [LevelConfig(0.8, 0.5, 110, 30, None, 0.5),
                   LevelConfig(0.8, 0.2, 40, 20, ('obj_no_change', 3), None)]

    hms = HMS(experiment, 2, config_list, 2, ('epochs', epochs), n_jobs=n_jobs, seed=seed, out_dir=DIR)

    future = client.submit(hms.run)

    create_exit_handler(future)

    logs = future.result()

    hms.log_summary_metrics(logs)

    input("Press any key to exit...")


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
