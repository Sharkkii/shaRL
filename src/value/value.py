#### Value function ####

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn

class BaseValue(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value_network,
        value_optimizer
    ):
        self.value_network = value_network
        self.value_optimizer = value_optimizer

    @abstractmethod
    def __call__(
        self,
        state
    ):
        return self.value_network(state)

    # @abstractmethod
    def copy(
        self
    ):
        return copy.deepcopy(self)

class Value(BaseValue):

    def __init__(
        self,
        value_network,
        value_optimizer
    ):
        super().__init__(
            value_network,
            value_optimizer
        )

    def __call__(
        self,
        state
    ):
        return self.value_network(state)

class BaseQValue(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        qvalue_network,
        qvalue_optimizer
    ):
        self.qvalue_network = qvalue_network
        self.qvalue_optimizer = qvalue_optimizer

    @abstractmethod
    def __call__(
        self,
        state,
        action
    ):
        return self.qvalue_network(state, action)
    
    # @abstractmethod
    def copy(
        self
    ):
        return copy.deepcopy(self)

class QValue(BaseValue):

    def __init__(
        self,
        qvalue_network,
        qvalue_optimizer
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
