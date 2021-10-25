#### Optimizer ####

from collections import defaultdict
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
        network
    ):
        self.optimizer = optimizer(network.parameters())
    
    def reset(
        self
    ):
        raise NotImplementedError
    
    def zero_grad(
        self
    ):
        # raise NotImplementedError
        self.optimizer.zero_grad()
    
    def setup(
        self
    ):
        raise NotImplementedError
    
    def step(
        self
    ):
        # raise NotImplementedError
        self.optimizer.step()
