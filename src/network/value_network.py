#### Value Network ####

import numpy as np
import torch
import torch.nn as nn

from .measure_network import BaseMeasureNetwork


class ValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        value_network
    ):
        assert(callable(value_network))
        self.network = value_network
    
    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass
    
    def __call__(
        self,
        state
    ):
        return self.network(state)

class DiscreteQValueNetwork(BaseMeasureNetwork):

    def __init__(
        self,
        qvalue_network
    ):
        assert(callable(qvalue_network))
        self.network = qvalue_network

    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass
    
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
        assert(callable(qvalue_network))
        self.network = qvalue_network

    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass
    
    def __call__(
        self,
        state,
        action
    ):
        return self.network(state, action)

DefaultValueNetwork = ValueNetwork
DefaultQValueNetwork = QValueNetwork = DiscreteQValueNetwork
DefaultDiscreteQValueNetwork = DiscreteQValueNetwork
DefaultContinuousQValueNetwork = ContinuousQValueNetwork
