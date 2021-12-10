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

    def __init__(
        self,
        optimizer,
        # network,
        **kwargs
    ):
        self.optimizer_class = optimizer
        self.optimizer = None
        self.kwargs = kwargs
    
    def reset(
        self
    ):
        raise NotImplementedError
    
    def zero_grad(
        self
    ):
        self.optimizer.zero_grad()
    
    def setup(
        self,
        network
    ):
        if (callable(network)):
            self.optimizer = self.optimizer_class(
                network.parameters(),
                **self.kwargs
            )
    
    def step(
        self
    ):
        self.optimizer.step()
