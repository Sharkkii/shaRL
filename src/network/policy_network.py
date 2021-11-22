#### Policy Network ####

import numpy as np
import torch
import torch.nn as nn

from .network import BaseNetwork

# PolicyNetwork
# .__init__
# .__call__
# .P (if necessary)
# .logP (if necessary)

class DiscretePolicyNetwork(BaseNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state: None)
    
    def __call__(
        self,
        state
    ):
        return self.network(state)

    # def predict(
    #     self,
    #     state
    # ):
    #     return self.network.predict(state)

class ContinuousPolicyNetwork(BaseNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state, noise: None)
    
    def __call__(
        self,
        state,
        action
    ):
        return self.network(state, action)

    # def predict(
    #     self,
    #     state,
    #     action
    # ):
    #     return self.network.predict(state, action)

PolicyNetwork = DiscretePolicyNetwork
        