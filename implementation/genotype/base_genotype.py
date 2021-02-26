from abc import ABC, abstractmethod
import numpy as np


class BaseGenotype(ABC):

    mut_prob: float
    mut_std: float

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def crossover(self, other_genotype: 'BaseGenotype') -> None:
        pass

    @abstractmethod
    def get_gene_array(self) -> np.ndarray:
        pass
