#### Policy ####

from abc import ABCMeta, abstractmethod
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
        self.policy_network = policy_network
        self.policy_optimizer = policy_optimizer

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        state,
        action = None
    ):
        raise NotImplementedError

class Policy(BasePolicy):
    
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().__init__(policy_network, policy_optimizer)

    def reset(
        self
    ):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def __call__(
        self,
        state,
        action = None
    ):
        raise NotImplementedError
