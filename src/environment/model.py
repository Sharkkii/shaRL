#### Model ####

from abc import ABCMeta, abstractmethod
import numpy as np


# base
class Model(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        self.state_space = None
        self.action_space = None
        self.observation_space = None
    
    @abstractmethod
    def setup(
        self,
        env
    ):
        # self.state_space = env.state_space
        self.action_space = env.action_space
        self.observation_space = env.observation_space

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
        self,
        n_times=1
    ):
        raise NotImplementedError


# default
class DefaultModel:

    def __init__(
        self
    ):
        self.env = None
        self.state_space = None
        self.action_space = None
        self.observation_space = None
    
    def setup(
        self,
        env
    ):
        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(
        self
    ):
        return self.env.reset()

    def step(
        self,
        action
    ):
        return self.env.step(action)

    def update(
        self,
        trajs,
        n_times=1
    ):
        pass
