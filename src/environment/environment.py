#### Environment ####

import math
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import gym
# from gym import spaces
# from gym.wrappers import TimeLimit


class BaseEnvironment(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        action
    ):
        raise NotImplementedError

class Environment(BaseEnvironment):

    def __init__(
        self
    ):
        pass
    
    def reset(
        self
    ):
        pass
    
    def step(
        self,
        action
    ):
        raise NotImplementedError

class GymEnvironment(BaseEnvironment):

    def __init__(
        self,
        name = ""
    ):
        self.env = None

    def reset(
        self
    ):
        self.env.reset()
    
    def step(
        self,
        action
    ):
        self.env.step(action)
