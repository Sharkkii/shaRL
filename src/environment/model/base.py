#### Model (Base Class) ####

from abc import ABC, abstractmethod


class ModelBase(ABC):

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
    # def score(self): raise NotImplementedError
    @abstractmethod
    def update(self): raise NotImplementedError
    # @abstractmethod
    # def train(self): raise NotImplementedError
    # @abstractmethod
    # def eval(self): raise NotImplementedError
    # @abstractmethod
    # def can_accept_action(self): raise NotImplementedError

    @property
    @abstractmethod
    def env(self): raise NotImplementedError
    @property
    @abstractmethod
    def interface(self): raise NotImplementedError
    @property
    @abstractmethod
    def configuration(self): raise NotImplementedError
    @property
    @abstractmethod
    def state(self): raise NotImplementedError
    @property
    @abstractmethod
    def state_space(self): raise NotImplementedError
    @property
    @abstractmethod
    def action_space(self): raise NotImplementedError
    @property
    @abstractmethod
    def observation_space(self): raise NotImplementedError


class EmptyModelBase(ModelBase):
    pass


class ApproximateModelBase(ModelBase):
    pass
