#### Critic (Base Class) ####

from abc import ABC, abstractmethod


class CriticBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    @abstractmethod
    def setup_with_actor(self): raise NotImplementedError
    @abstractmethod
    def epochwise_preprocess(self): raise NotImplementedError
    @abstractmethod
    def epochwise_postprocess(self): raise NotImplementedError
    @abstractmethod
    def stepwise_preprocess(self): raise NotImplementedError
    @abstractmethod
    def stepwise_postprocess(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    # @abstractmethod
    # def __call__(self): raise NotImplementedError
    @abstractmethod
    def update(self): raise NotImplementedError
    @abstractmethod
    def update_value(self): raise NotImplementedError
    @abstractmethod
    def update_qvalue(self): raise NotImplementedError
    @abstractmethod
    def train(self): raise NotImplementedError
    @abstractmethod
    def eval(self): raise NotImplementedError

    @property
    @abstractmethod
    def interface(self): raise NotImplementedError
    @property
    @abstractmethod
    def configuration(self): raise NotImplementedError
    @property
    @abstractmethod
    def value(self): raise NotImplementedError
    @property
    @abstractmethod
    def qvalue(self): raise NotImplementedError


class DiscreteControlCriticBase(CriticBase):
    pass


class ContinuousControlCriticBase(CriticBase):
    pass
