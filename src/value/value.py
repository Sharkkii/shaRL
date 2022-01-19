#### Value function ####

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn

from ..network import PseudoMeasureNetwork
from ..network import BaseMeasureNetwork
from ..network import cast_to_measure_network
from ..optimizer import Optimizer

class BaseValue(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value_network = None,
        value_optimizer = None
    ):
        self.value_network = cast_to_measure_network(value_network)
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
        if (isinstance(self.value_network, PseudoMeasureNetwork) and isinstance(value_network, BaseMeasureNetwork)):
            self.value_network = value_network
        if ((self.value_optimizer is None) and (type(value_optimizer) is Optimizer)):
            self.value_optimizer = value_optimizer
            self.value_optimizer.setup(
                network = self.value_network
            )
        print(f"Value.setup: { self.value_network } & { self.value_optimizer }")

    def train(
        self
    ):
        self.value_network.train()
    
    def eval(
        self
    ):
        self.value_network.eval()

    def save(
        self,
        path_to_value_network
    ):
        self.value_network.save(path_to_value_network)
    
    def load(
        self,
        path_to_value_network
    ):
        self.value_network.load(path_to_value_network)

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

class PseudoValue(BaseValue):

    @classmethod
    def __raise_exception(cls):
        raise Exception("`PseudoValue` cannot be used as a value function.")

    def __init__(
        self,
        value_network = None,
        value_optimizer = None
    ):
        pass

    def reset(
        self
    ):
        pass

    def setup(
        self,
        value_network = None,
        value_optimizer = None
    ):
        pass

    def train(
        self
    ):
        pass

    def eval(
        self
    ):
        pass

    def __call__(
        self,
        state
    ):
        PseudoValue.__raise_exception()
    
    def save(
        self,
        path_to_value_network
    ):
        PseudoValue.__raise_exception()

    def load(
        self,
        path_to_value_network
    ):
        PseudoValue.__raise_exception()

class BaseQValue(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        self.qvalue_network = cast_to_measure_network(qvalue_network)
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
        if ((isinstance(self.qvalue_network, PseudoMeasureNetwork)) and isinstance(qvalue_network, BaseMeasureNetwork)):
            self.qvalue_network = qvalue_network
        if ((self.qvalue_optimizer is None) and (type(qvalue_optimizer) is Optimizer)):
            self.qvalue_optimizer = qvalue_optimizer
            self.qvalue_optimizer.setup(
                network = self.qvalue_network
            )
            print(f"QValue.setup: { self.qvalue_network } & { self.qvalue_optimizer }")
    
    def train(
        self
    ):
        self.qvalue_network.train()
    
    def eval(
        self
    ):
        self.qvalue_network.eval()
    
    def save(
        self,
        path_to_qvalue_network
    ):
        self.qvalue_network.save(path_to_qvalue_network)
    
    def load(
        self,
        path_to_qvalue_network
    ):
        self.qvalue_network.load(path_to_qvalue_network)
    
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

class PseudoQValue(BaseQValue):

    @classmethod
    def __raise_exception(cls):
        raise Exception("`PseudoQValue` cannot be used as a qvalue function.")

    def __init__(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        pass

    def reset(
        self
    ):
        pass

    def setup(
        self,
        qvalue_network = None,
        qvalue_optimizer = None
    ):
        pass

    def train(
        self
    ):
        pass

    def eval(
        self
    ):
        pass

    def __call__(
        self,
        state,
        action = None
    ):
        PseudoQValue.__raise_exception()
    
    def save(
        self,
        path_to_qvalue_network
    ):
        PseudoQValue.__raise_exception()

    def load(
        self,
        path_to_qvalue_network
    ):
        PseudoQValue.__raise_exception()
