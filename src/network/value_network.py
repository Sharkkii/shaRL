#### Value Network ####

import numpy as np
import torch
import torch.nn as nn

from ..common import AgentInterface
from .measure_network import BaseMeasureNetwork
from .network import PseudoNetwork


class ValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        value_network = None,
        interface = None,
        use_default = False
    ):
        if (use_default):
            if (value_network is not None):
                raise ValueError("`value_network` must be None if `use_default = True`")
            if (type(interface) is not AgentInterface):
                raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`")
            value_network = PseudoNetwork()
            
        super().__init__(
            network = value_network
        )
    
    def reset(
        self
    ):
        pass

    def setup(
        self,
        value_network = None
    ):
        super().setup(
            network = value_network
        )

    def __call__(
        self,
        state
    ):
        return self.network(state)

class DiscreteQValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        qvalue_network = None,
        interface = None,
        use_default = False
    ):
        if (use_default):
            if (qvalue_network is not None):
                raise ValueError("`qvalue_network` must be None if `use_default = True`")
            if (type(interface) is not AgentInterface):
                raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`")
            qvalue_network = PseudoNetwork()

        super().__init__(
            network = qvalue_network
        )

    def reset(
        self
    ):
        pass

    def setup(
        self,
        qvalue_network = None
    ):
        super().setup(
            network = qvalue_network
        )
    
    def __call__(
        self,
        state,
        action = None
    ):
        assert(action is None)
        return self.network(state)

class ContinuousQValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        qvalue_network,
        interface = None,
        use_default = False
    ):
        super().__init__(
            network = qvalue_network
        )

    def reset(
        self
    ):
        pass

    def setup(
        self,
        qvalue_network = None
    ):
        super().setup(
            network = qvalue_network
        )
    
    def __call__(
        self,
        state,
        action
    ):
        return self.network(state, action)

QValueNetwork = DiscreteQValueNetwork
