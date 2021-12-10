#### Value function ####

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn

from ..optimizer import Optimizer

class BaseValue(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value_network = None,
        value_optimizer = None
    ):
        self.value_network = value_network
        self.value_optimizer = value_optimizer

    @abstractmethod
    def __call__(
        self,
        state
    ):
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        value_network = None,
        value_optimizer = None
    ):
        if ((self.value_network is None) and callable(value_network)):
            self.value_network = value_network
        if ((self.value_optimizer is None) and (type(value_optimizer) is Optimizer)):
            self.value_optimizer = value_optimizer
            
        if (self.value_optimizer is not None):
            self.value_optimizer.setup(
                network = self.value_network
            )

    def train(
        self
    ):
        self.value_network.train()
    
    def eval(
        self
    ):
        self.value_network.eval()

    # @abstractmethod
    def copy(
        self
    ):
        return copy.deepcopy(self)

class Value(BaseValue):

    def __init__(
        self,
        value_network = None,
        value_optimizer = None
    ):
        super().__init__(
            value_network = value_network,
            value_optimizer = value_optimizer
        )

    def __call__(
        self,
        state
    ):
        return self.value_network(state)
    
    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        value_network = None,
        value_optimizer = None
    ):
        super().setup(
            value_network = value_network,
            value_optimizer = value_optimizer
        )

class BaseQValue(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        self.qvalue_network = qvalue_network
        self.qvalue_optimizer = qvalue_optimizer

    @abstractmethod
    def __call__(
        self,
        state,
        action = None
    ):
        raise NotImplementedError

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        if ((self.qvalue_network is None) and callable(qvalue_network)):
            self.qvalue_network = qvalue_network
        if ((self.qvalue_optimizer is None) and (type(qvalue_optimizer) is Optimizer)):
            self.qvalue_optimizer = qvalue_optimizer
        
        if (self.qvalue_optimizer is not None):
            self.qvalue_optimizer.setup(
                network = self.qvalue_network
            )
    
    def train(
        self
    ):
        self.qvalue_network.train()
    
    def eval(
        self
    ):
        self.qvalue_network.eval()
    
    # @abstractmethod
    def copy(
        self
    ):
        return copy.deepcopy(self)

class DiscreteQValue(BaseQValue):

    def __init__(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        super().__init__(
            qvalue_network,
            qvalue_optimizer
        )

    def __call__(
        self,
        state,
        action = None
    ):
        assert(action is None)
        return self.qvalue_network(state)

    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        super().setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )

class ContinuousQValue(BaseQValue):

    def __init__(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        super().__init__(
            qvalue_network,
            qvalue_optimizer
        )

    def __call__(
        self,
        state,
        action
    ):
        return self.qvalue_network(state, action)

    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        super().setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )

QValue = DiscreteQValue