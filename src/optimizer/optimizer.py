#### Optimizer ####

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.optim

from ..network import BaseMeasureNetwork
from ..network import PseudoMeasureNetwork

class BaseOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        self.optimizer = None
        self.network = None
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def zero_grad(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def setup(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self
    ):
        raise NotImplementedError

class Optimizer(BaseOptimizer):

    def check_whether_available(f):
        def wrapper(self, *args, **kwargs):
            if (self.optimizer is None):
                raise Exception(f"Call `Optimizer.setup` before using `Optimizer.{ f.__name__ }`")
            return f(self, *args, **kwargs)
        return wrapper

    def __init__(
        self,
        optimizer,
        network = None,
        **kwargs
    ):
        self.optimizer_factory = optimizer # should be implemented as MetaClass
        self.optimizer = None
        self.network = None
        self._is_available = False

        self.setup(
            network = network,
            **kwargs
        )

    def reset(
        self
    ):
        raise NotImplementedError
    
    @check_whether_available
    def zero_grad(
        self
    ):
        self.optimizer.zero_grad()
    
    def setup(
        self,
        network,
        **kwargs
    ):
        if (isinstance(network, BaseMeasureNetwork)):
            self.optimizer = self.optimizer_factory(
                network.parameters(),
                **kwargs
            )
            self.network = network
            self._become_available()
            print(f"Optimizer.setup: { self.optimizer_factory }({ network })")

    @property
    def is_available(
        self
    ):
        return self._is_available

    def _become_available(
        self
    ):
        self._is_available = True

    def _become_unavailable(
        self
    ):
        self._is_available = False

    @check_whether_available
    def step(
        self
    ):
        self.optimizer.step()

    @check_whether_available
    def clip_grad_value(
        self,
        value = 1.0
    ):
        for parameter in self.network.parameters():
            torch.nn.utils.clip_grad_value_(parameter.grad, value)
    
    @check_whether_available
    def clip_grad_norm(
        self,
        value = 1.0
    ):
        for parameter in self.network.parameters():
            torch.nn.utils.clip_grad_norm_(parameter.grad, value, norm_type=2)
