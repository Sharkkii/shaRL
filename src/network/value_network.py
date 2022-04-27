#### Value Network ####

import numpy as np
import torch
import torch.nn as nn

from .measure_network import BaseMeasureNetwork


class ValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        value_network = None
    ):
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
        qvalue_network = None
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
        action = None
    ):
        assert(action is None)
        return self.network(state)

class ContinuousQValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        qvalue_network
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
