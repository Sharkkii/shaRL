#### Optimizer ####

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim


# base
class Optimizer(metaclass=ABCMeta):

    def __init__(
        self,
        params=None,
        defaults={}
    ):
        self.params = params
        self.defaults = defaults
        self.state = {}

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        loss
    ):
        raise NotImplementedError

    @abstractmethod
    def step(
        self
    ):
        raise NotImplementedError


class DefaultOptimizer(Optimizer):

    def __init__(
        self,
        constructor=None,
        target=None,
        options={}
    ):
        assert(constructor is not None)
        assert(target is not None)
        self.target = target.parameters()
        self.optimizer = constructor(self.target, **options)
    
    def reset(self):
        return self.optimizer.zero_grad()
    
    @torch.no_grad()
    def step(self):
        return self.optimizer.step()

    # alias
    def zero_grad(self):
        return self.reset()


# optimizer-wrapper
class OptWrapper(Optimizer):
    def __init__(
        self,
        optimizer,
        params,
        **kwargs,
    ):
        defaults = dict(kwargs)
        super(OptWrapper, self).__init__(params, defaults)
        self.optimizer = optimizer(
            params=params,
            **kwargs
        )

    def reset(self):
        self.optimizer.zero_grad()

    # alias
    def zero_grad(self):
        self.reset()
    
    def setup(self, loss):
        loss.backward()
    
    @torch.no_grad()
    def step(self):
        self.optimizer.step()


# optimizer for TabularNetwork
class TabularOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr
    ):
        # NOTE: params expects only one element
        assert(len(list(params)) == 1)
        defaults = { "lr": lr }
        super(TabularOptimizer, self).__init__(
            params=params,
            defaults=defaults
        )

    def reset(self):
        self.state["grad"] = None

    def setup(self, loss):
        assert(loss is not None)
        self.state["grad"] = loss

    @torch.no_grad()
    def step(self):
        param = self.params[0]
        lr = self.defaults["lr"]
        grad = self.state["grad"]
        param.add_(grad, alpha=-lr)



