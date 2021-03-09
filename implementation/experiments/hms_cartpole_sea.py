from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.gym_env import GymEnv
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_nn import IndividualNN
from . import run_arg_parser
from typing import Tuple
from numpy.random import Generator, SeedSequence
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import numpy as np
import sys
import os
import signal

LENGTH = 58


class ExperimentCartPole(BaseExperiment):

    def __init__(self, encoding: str = 'fixed'):
        super().__init__(encoding)

    def create_individual(self, level_config: LevelConfig, rng: Generator, noise=None) -> BaseIndividual:
        gen = self.genotype(level_config.mut_prob, level_config.mut_std, LENGTH, rng, noise)
        return IndividualNN(gen, [(4, 8), (8,), (8, 2), (2,)])

    def evaluate_individual(self, deme_id, individual_id, individual: BaseIndividual, seed: SeedSequence) -> Tuple[int, int, float]:
        env = GymEnv(env_name="CartPole-v0", seed=seed)
        return env.run_evaluation(individual, deme_id, individual_id)


def run_experiment(seed, n_jobs, epochs):

    cluster = SLURMCluster(queue=os.environ['PARTITION'],
                           project=os.environ['GRANT'],
                           cores=1,
                           processes=2,
                           memory='6GB',
                           walltime='00:06:00')

    cluster.scale(n_jobs)

    client = Client(cluster)  # n_workers=n_jobs

    experiment = ExperimentCartPole()

    config_list = [LevelConfig(0.8, 0.5, 150, 30, None, None)]

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', epochs), n_jobs=n_jobs, seed=seed)

    future = client.submit(hms.run)

    def handler(signum, _):
        if __name__ == '__main__':
            future.cancel()
            client.shutdown()
            print('Stopping experiment early. Saving scores...')
            sys.exit()

    signal.signal(signal.SIGINT, handler)

    logs = future.result()

    hms.log_summary_metrics(logs)

    input("Press any key to exit...")


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
