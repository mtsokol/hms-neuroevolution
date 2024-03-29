from abc import ABC, abstractmethod
from ..genotype.base_individual import BaseIndividual
from numpy.random import Generator, SeedSequence
from ..evolution.hms.config import LevelConfig
from ..genotype.genotype_fixed_len import GenotypeFixedLen
from ..genotype.genotype_var_len import GenotypeVarLen


class BaseExperiment(ABC):

    def __init__(self, encoding):
        if encoding == 'fixed':
            self.genotype = GenotypeFixedLen
        elif encoding == 'var':
            self.genotype = GenotypeVarLen
        else:
            raise Exception('Invalid encoding type')

    @abstractmethod
    def create_individual(self, level_config: LevelConfig, rng: Generator) -> BaseIndividual:
        pass

    @abstractmethod
    def evaluate_individual(self, individual: BaseIndividual, individual_id, deme_id, seed: SeedSequence) -> float:
        pass
