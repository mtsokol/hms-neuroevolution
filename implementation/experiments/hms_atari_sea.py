from ..evolution.hms.hms import HMS
from ..evolution.hms.config import LevelConfig
from ..environment.atari_env import AtariEnv
from .base_experiment import BaseExperiment
from ..genotype.base_individual import BaseIndividual
from ..genotype.individual_conv import IndividualConv
from ..visualization import make_dir
from typing import Tuple
from numpy.random import Generator, SeedSequence
import numpy as np
from . import run_arg_parser, create_client, create_exit_handler

LENGTH = 1687218


class ExperimentAtari(BaseExperiment):

    def __init__(self, encoding: str = 'var'):
        super().__init__(encoding)

    def create_individual(self, level_config: LevelConfig, rng: Generator) -> BaseIndividual:
        gen = self.genotype(level_config.mut_prob, level_config.mut_std, LENGTH, rng)
        return IndividualConv(gen, [(8, 8, 1, 32), (32,), (4, 4, 32, 64), (64,), (3, 3, 64, 64),
                                    (64,), (3136, 512), (512,), (512, 18), (18,)])

    def evaluate_individual(self, deme_id, individual_id, individual: BaseIndividual, seed: SeedSequence) -> Tuple[
        int, int, float]:
        env = AtariEnv(env_name="Frostbite-v0", seed=seed)
        return env.run_evaluation(individual, deme_id, individual_id)


def run_experiment(seed, n_jobs, epochs):

    client = create_client(n_jobs)

    experiment = ExperimentAtari()

    config_list = [LevelConfig(1.0, 0.002, 1000, 20, None, None)]

    out_dir = make_dir()

    hms = HMS(experiment, 1, config_list, np.inf, ('epochs', epochs), n_jobs=n_jobs, seed=seed, out_dir=out_dir, client=client)

    create_exit_handler(client)

    logs = hms.run()

    hms.log_summary_metrics(logs)

    input("Press any key to exit...")


if __name__ == '__main__':

    run_experiment(*run_arg_parser())
