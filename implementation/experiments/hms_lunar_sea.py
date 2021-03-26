from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.gym_env import GymEnv
from ..visualization import make_dir
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_nn import IndividualNN
from . import run_arg_parser, create_client, create_exit_handler
from typing import Tuple
from numpy.random import Generator, SeedSequence
import numpy as np

LENGTH = 18180


class ExperimentLunarLander(BaseExperiment):

    def __init__(self, encoding: str = 'fixed'):
        super().__init__(encoding)

    def create_individual(self, level_config: LevelConfig, rng: Generator) -> BaseIndividual:
        gen = self.genotype(level_config.mut_prob, level_config.mut_std, LENGTH, rng)
        return IndividualNN(gen, [(8, 128), (128,), (128, 128), (128,), (128, 4), (4,)])

    def evaluate_individual(self, deme_id, individual_id, individual: BaseIndividual, seed: SeedSequence) -> Tuple[int, int, float]:
        env = GymEnv(env_name="LunarLander-v2", seed=seed)
        return env.run_evaluation(individual, deme_id, individual_id)


def run_experiment(seed, n_jobs, epochs):

    client = create_client(n_jobs)

    experiment = ExperimentLunarLander()

    config_list = [LevelConfig(0.9, 0.2, 450, 35, None, None)]

    out_dir = make_dir()

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', epochs), n_jobs=n_jobs, seed=seed, out_dir=out_dir)

    future = client.submit(hms.run)

    create_exit_handler(future, client)

    logs = future.result()

    hms.log_summary_metrics(logs)

    input("Press any key to exit...")


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
