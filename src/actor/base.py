#### Actor (Base Class) ####

from abc import ABC, abstractmethod


class ActorBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    @abstractmethod
    def setup_with_critic(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __call__(self): raise NotImplementedError
    @abstractmethod
    def choose_action(self): raise NotImplementedError
    @abstractmethod
    def update(self): raise NotImplementedError
    @abstractmethod
    def update_policy(self): raise NotImplementedError
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
    def policy(self): raise NotImplementedError
    @property
    @abstractmethod
    def can_pointwise_estimate(self): raise NotImplementedError
    @property
    @abstractmethod
    def can_density_estimate(self): raise NotImplementedError


class DiscreteControlActorBase(ActorBase):
    pass


class ContinuousControlActorBase(ActorBase):
    pass


class GoalConditionedActorBase(ActorBase):
    pass
