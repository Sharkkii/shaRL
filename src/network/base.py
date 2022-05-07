from abc import ABCMeta, abstractmethod

class BaseNetwork(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        interface = None
    ):
        self.interface = interface
        self._is_available = True

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError
