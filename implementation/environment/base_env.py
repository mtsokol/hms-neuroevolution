from abc import ABC, abstractmethod


class BaseEnv(ABC):

    @abstractmethod
    def env_step(self):
        raise NotImplementedError
