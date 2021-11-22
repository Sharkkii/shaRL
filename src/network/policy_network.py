#### Policy Network ####

import numpy as np
import torch
import torch.nn as nn

from .network import BasePolicyNetwork


class DiscretePolicyNetwork(BasePolicyNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state: None)

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

    def P(
        self,
        state
    ):
        p = self.policy_network(state)
        return p

    def logP(
        self,
        state
    ):
        log_p = torch.log(self.P(state))
        return log_p

class ContinuousPolicyNetwork(BasePolicyNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state, noise: None)

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

    def P(
        self,
        state,
        action
    ):
        p = self.policy_network(state, action)
        return p

    def logP(
        self,
        state,
        action
    ):
        log_p = torch.log(self.P(state, action))
        return log_p

PolicyNetwork = DiscretePolicyNetwork
        