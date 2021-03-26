from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..visualization import make_dir
from .hms_atari_sea import ExperimentAtari
from . import run_arg_parser, create_client, create_exit_handler


def run_experiment(seed, n_jobs, epochs):

    client = create_client(n_jobs)

    experiment = ExperimentAtari()

    config_list = [LevelConfig(1.0, 0.05, 600, 45, None, 0.5),
                   LevelConfig(1.0, 0.002, 150, 20, ('obj_no_change', 3), None)]

    out_dir = make_dir()

    hms = HMS(experiment, 2, config_list, 3, ('epochs', epochs), n_jobs=n_jobs, seed=seed, out_dir=out_dir)

    future = client.submit(hms.run)

    create_exit_handler(future, client)

    logs = future.result()

    hms.log_summary_metrics(logs)

    input("Press any key to exit...")


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
