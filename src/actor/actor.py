#### Actor ####

import os
import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from gym.spaces import Box, Discrete

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from policy import Policy


class BaseActor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy = None
    ):
        raise NotImplementedError
        # self.policy = policy
        # self.behavior_policy = None
        # self.target_policy = None

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
        self.policy.setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )

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

    # @abstractmethod
    # def update_behavior_policy(
    #     self,
    #     trajectory
    # ):
    #     raise NotImplementedError

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
        self.policy = Policy() if policy is None else policy
        self.target_policy = self.policy.copy()
    
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
    
    def choose_action(
        self,
        state,
        action_space
    ):
        action = self.policy(state)
        if (action is None):
            action = action_space.sample()
        else:
            action = torch.argmax(action) if (type(action_space) is Discrete) else action
        return action
    
    def update_policy(
        self,
        critic,
        trajectory
    ):
        pass

    # def update_behavior_policy(
    #     self,
    #     trajectory
    # ):
    #     pass

    def update_target_policy(
        self,
        critic,
        trajectory
    ):
        pass

    def update(
        self,
        critic,
        trajectory,
        n_times = 1
    ):
        for _ in range(n_times):
            self.update_policy(critic, trajectory)
            self.update_target_policy(critic, trajectory)
