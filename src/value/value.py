#### Value function ####

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn

from ..const import SpaceType
from ..common import AgentInterface
from ..common import Component
from ..network import ValueNetwork
from ..network import QValueNetwork
from ..network import DiscreteQValueNetwork
from ..network import ContinuousQValueNetwork
from ..optimizer import MeasureOptimizer

class BaseValue(Component, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value_network = None,
        value_optimizer = None,
        interface = None,
        use_default = False
    ):
        Component.__init__(self)
        if (use_default):
            if (not ((value_network is None) and (value_optimizer is None))):
                raise ValueError("`value_network` & `value_optimizer` must be None if `use_default = True`")
            value_network = ValueNetwork(
                interface = interface,
                use_default = True
            )
            value_optimizer = MeasureOptimizer(torch.optim.Adam, lr=1e-3)

        self.value_network = None
        self.value_optimizer = None
        self.setup(
            value_network = value_network,
            value_optimizer = value_optimizer
        )

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
        if ((value_network is not None) and (value_optimizer is not None)):
            self.value_network = value_network
            self.value_optimizer = value_optimizer
            self.value_optimizer.setup(
                network = self.value_network
            )
            self._become_available()
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
        value_optimizer = None,
        interface = None,
        use_default = False
    ):
        super().__init__(
            value_network = value_network,
            value_optimizer = value_optimizer,
            interface = interface,
            use_default = use_default
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

class BaseQValue(Component, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        qvalue_network = None,
        qvalue_optimizer = None,
        interface = None,
        use_default = False
    ):
        Component.__init__(self)
        if (use_default):
            if (not ((qvalue_network is None) and (qvalue_optimizer is None))):
                raise ValueError("`qvalue_network` & `qvalue_optimizer` must be None if `use_default = True`")
            if (type(interface) is not AgentInterface):
                raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`")
            if (interface.tout is SpaceType.DISCRETE):
                qvalue_network = DiscreteQValueNetwork(
                    interface = interface,
                    use_default = True
                )
            elif (interface.tout is SpaceType.CONTINUOUS):
                qvalue_network = ContinuousQValueNetwork(
                    interface = interface,
                    use_default = True
                )
            else:
                raise ValueError("invalid interface")
            qvalue_optimizer = MeasureOptimizer(torch.optim.Adam, lr=1e-3)

        self.qvalue_network = None
        self.qvalue_optimizer = None
        self.setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )

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
        if ((qvalue_network is not None) and (qvalue_optimizer is not None)):
            self.qvalue_network = qvalue_network
            self.qvalue_optimizer = qvalue_optimizer
            self.qvalue_optimizer.setup(
                network = self.qvalue_network
            )
            self._become_available()
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
        qvalue_optimizer = None,
        interface = None,
        use_default = False
    ):
        super().__init__(
            qvalue_network,
            qvalue_optimizer,
            interface = interface,
            use_default = use_default
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
        qvalue_optimizer = None,
        interface = None,
        use_default = False
    ):
        super().__init__(
            qvalue_network,
            qvalue_optimizer,
            interface = interface,
            use_default = use_default
        )

    def __call__(
        self,
        state,
        action = None
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
