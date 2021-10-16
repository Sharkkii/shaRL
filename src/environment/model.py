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

class Model(BaseModel):

    def __init__(
        self
    ):
        # self.state_space = None
        # self.action_space = None
        # self.observation_space = None
        pass
    
    def setup(
        self,
        env
    ):
        # self.state_space = env.state_space
        # self.action_space = env.action_space
        # self.observation_space = env.observation_space
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

    def update(
        self,
        n_times = 1
    ):
        pass
