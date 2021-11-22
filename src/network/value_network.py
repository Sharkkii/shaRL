#### Value Network ####

import numpy as np
import torch
import torch.nn as nn

from .network import BaseNetwork


class ValueNetwork(BaseNetwork):

    def __init__(
        self,
        value_network
    ):
        self.network = value_network if callable(value_network) else (lambda state: None)
    
    def __call__(
        self,
        state
    ):
        return self.network(state)

class DiscreteQValueNetwork(BaseNetwork):

    def __init__(
        self,
        qvalue_network
    ):
        self.network = qvalue_network if callable(qvalue_network) else (lambda state: None)
    
    def __call__(
        self,
        state,
        action = None
    ):
        assert(action is None)
        return self.network(state)

class ContinuousQValueNetwork(BaseNetwork):

    def __init__(
        self,
        qvalue_network
    ):
        self.network = qvalue_network if callable(qvalue_network) else (lambda state, action: None)
    
    def __call__(
        self,
        state,
        action
    ):
        return self.network(state, action)

QValueNetwork = DiscreteQValueNetwork