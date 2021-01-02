from abc import ABC, abstractmethod


class BaseEnv(ABC):

    @abstractmethod
    def xd(self):
        raise NotImplementedError
