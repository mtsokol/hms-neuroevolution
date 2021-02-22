from abc import ABC, abstractmethod
from ..genotype.base_individual import BaseIndividual
import numpy as np
from numpy.random import Generator, SeedSequence
from ..evolution.hms.config import LevelConfig
from ..genotype.genotype_fixed_len import GenotypeFixedLen
from ..genotype.genotype_var_len import GenotypeVarLen
from typing import Optional


class BaseExperiment(ABC):

    def __init__(self, encoding):
        if encoding == 'fixed':
            self.genotype = GenotypeFixedLen
        elif encoding == 'var':
            self.genotype = GenotypeVarLen
        else:
            raise Exception('Invalid encoding type')

    @abstractmethod
    def create_individual(self, level_config: LevelConfig, rng: Generator, noise: Optional[np.ndarray]) -> BaseIndividual:
        pass

    @abstractmethod
    def evaluate_individual(self, individual: BaseIndividual, individual_id, deme_id, seed: SeedSequence) -> np.float:
        pass
