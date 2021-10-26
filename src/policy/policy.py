#### Policy ####

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn


class BasePolicy(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        self.policy_network = policy_network if callable(policy_network) else (lambda state: None)
        self.policy_optimizer = policy_optimizer

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
        policy_network = None,
        policy_optimizer = None
    ):
        if callable(policy_network):
            self.policy_network = policy_network
        if (policy_optimizer is not None):
            self.policy_optimizer = policy_optimizer
    
    # @abstractmethod
    def copy(
        self
    ):
        return copy.deepcopy(self)

class Policy(BasePolicy):
    
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )

    def __call__(
        self,
        state
    ):
        return self.policy_network(state)

    def predict(
        self,
        state
    ):
        return self.policy_network.predict(state)
        
    def reset(
        self
    ):
        pass

    def setup(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )

