#### Environment ####

# import math
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import torch
import gym
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

    @abstractmethod
    def update(
        self
    ):
        raise NotImplementedError

class Environment(BaseEnvironment):

    def __init__(
        self
    ):
        self.state_space = None
        self.action_space = None
        self.observation_space = None
        self.state = None
    
    def reset(
        self
    ):
        observation = self.state = None
        return observation
    
    def step(
        self,
        action
    ):
        observation = None
        reward = None
        done = True
        info = None
        return observation, reward, done, info
    
    def update(
        self
    ):
        warnings.warn("`update` cannot be used for an Environment instance.")

class GymEnvironment(Environment):

    def __init__(
        self,
        name = ""
    ):
        self.env = gym.make(name)
        self.state_space = self.env.observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state = None
        
    def reset(
        self
    ):
        observation = self.state = self.env.reset()
        return observation
    
    def step(
        self,
        action
    ):
        return self.env.step(action)
