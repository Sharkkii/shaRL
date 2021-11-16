#### Model ####

from abc import ABCMeta, abstractmethod
import numpy as np
from .environment import BaseEnvironment


class BaseModel(BaseEnvironment, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def setup(
        self,
        env
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
        self,
        n_times = 1
    ):
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        trajectory
    ):
       raise NotImplementedError

class Model(BaseModel):

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
    
    def setup(
        self,
        env
    ):
        self.observation_space = self.state_space = env.observation_space
        self.action_space = env.action_space
    
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
        self,
        n_times = 1
    ):
        pass

    def score(
        self,
        trajectory
    ):
        score_dictionary = {}
        return score_dictionary
