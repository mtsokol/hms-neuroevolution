from abc import ABC, abstractmethod
from ..genotype.base_individual import BaseIndividual
import numpy as np
from numpy.random import Generator, SeedSequence
from ..evolution.hms.config import LevelConfig


class BaseExperiment(ABC):

    @abstractmethod
    def create_individual(self, level_config: LevelConfig, rng: Generator) -> BaseIndividual:
        pass

    @abstractmethod
    def evaluate_individual(self, individual: BaseIndividual, individual_id, deme_id, seed: SeedSequence) -> np.float:
        pass
