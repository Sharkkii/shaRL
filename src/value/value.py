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
        value_network = None,
        value_optimizer = None
    ):
        self.value_network = value_network if callable(value_network) else (lambda state: None)
        self.value_optimizer = value_optimizer

    @abstractmethod
    def __call__(
        self,
        state
    ):
        raise NotImplementedError
        # return self.value_network(state)
    
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
        if callable(value_network):
            self.value_network = value_network
        if (value_optimizer is not None):
            self.value_optimizer = value_optimizer
        self.value_optimizer.setup(
            network = self.value_network
        )

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
        self.qvalue_network = qvalue_network if callable(qvalue_network) else (lambda state, action=None: None)
        self.qvalue_optimizer = qvalue_optimizer

    @abstractmethod
    def __call__(
        self,
        state,
        action = None
    ):
        # return self.qvalue_network(state, action)
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
        if callable(qvalue_network):
            self.qvalue_network = qvalue_network
        if (qvalue_optimizer is not None):
            self.qvalue_optimizer = qvalue_optimizer
        self.qvalue_optimizer.setup(
            network = self.qvalue_network
        )
    
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