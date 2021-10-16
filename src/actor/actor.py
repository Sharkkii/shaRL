#### Actor ####

import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import torch


class BaseActor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy = None,
        optim_policy = None
    ):
        self.policy = policy
        self.behavior_policy = None
        self.target_policy = None
        self.optim_policy = optim_policy

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self,
        state
    ):
        raise NotImplementedError

    @abstractmethod
    def update_policy(
        self,
        trajectory = None
    ):
        raise NotImplementedError

    @abstractmethod
    def update_behavior_policy(
        self,
        trajectory
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_policy(
        self,
        trajectory
    ):
        raise NotImplementedError
    
    @abstractmethod
    def update(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError

class Actor(BaseActor):

    def __init__(
        self,
        policy = None
    ):
        self.policy = policy
    
    def reset(
        self
    ):
        raise NotImplementedError
    
    def setup(
        self
    ):
        raise NotImplementedError
    
    def choose_action(
        self,
        state
    ):
        raise NotImplementedError
    
    def update_policy(
        self,
        trajectory
    ):
        pass

    def update_behavior_policy(
        self,
        trajectory
    ):
        pass

    def update_target_policy(
        self,
        trajectory
    ):
        pass

    def update(
        self,
        trajectory,
        n_times = 1
    ):
        for _ in range(n_times):
            self.update_behavior_policy(trajectory)
            self.update_target_policy(trajectory)
