#### Environment (Base Class) ####

from abc import ABC, abstractmethod


class EnvironmentBase(ABC):
    
    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    @abstractmethod
    def reset(self): raise NotImplementedError
    @abstractmethod
    def step(self): raise NotImplementedError
    @abstractmethod
    def sample(self): raise NotImplementedError
    @abstractmethod
    def score(self): raise NotImplementedError
    @abstractmethod
    def can_accept_action(self): raise NotImplementedError

    @property
    @abstractmethod
    def interface(self): raise NotImplementedError
    @property
    @abstractmethod
    def configuration(self): raise NotImplementedError
    @property
    @abstractmethod
    def state_space(self): raise NotImplementedError
    @property
    @abstractmethod
    def action_space(self): raise NotImplementedError
    @property
    @abstractmethod
    def observation_space(self): raise NotImplementedError
    @property
    @abstractmethod
    def state(self): raise NotImplementedError


class GymEnvironmentBase(EnvironmentBase):
    
    @property
    @abstractmethod
    def env(self): raise NotImplementedError


class GoalReachingTaskEnvironmentBase(EnvironmentBase):
    
    @property
    @abstractmethod
    def goal_space(self): raise NotImplementedError
    @property
    @abstractmethod
    def goal_state(self): raise NotImplementedError
