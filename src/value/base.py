#### Value (Base Class) ####

from abc import ABC, abstractmethod


class ValueBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def setup_with_policy(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __call__(self): raise NotImplementedError
    # @abstractmethod
    # def update(self): raise NotImplementedError
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
    def value_network(self): raise NotImplementedError
    @property
    @abstractmethod
    def value_optimizer(self): raise NotImplementedError


class EmptyValueBase(ValueBase):
    pass


class QValueBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def setup_with_policy(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __call__(self): raise NotImplementedError
    # @abstractmethod
    # def update(self): raise NotImplementedError
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
    def qvalue_network(self): raise NotImplementedError
    @property
    @abstractmethod
    def qvalue_optimizer(self): raise NotImplementedError


class EmptyQValueBase(ValueBase):
    pass


class DiscreteQValueBase(QValueBase):
    pass


class ContinuousQValueBase(QValueBase):
    pass
