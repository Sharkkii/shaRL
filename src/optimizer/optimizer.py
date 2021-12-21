#### Optimizer ####

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.optim


class BaseOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        raise NotImplementedError
    
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
            f(self, *args, **kwargs)
        return wrapper

    def __init__(
        self,
        optimizer,
        **kwargs
    ):
        self.optimizer_factory = optimizer
        self.optimizer = None
        self.network = None
        self.kwargs = kwargs

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
        network
    ):
        if (callable(network)):
            self.optimizer = self.optimizer_factory(
                network.parameters(),
                **self.kwargs
            )
            self.network = network
            print(f"Optimizer.setup: { self.optimizer_factory }({ network })")

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
