#### Policy (Base Class) ####

from abc import ABC, abstractmethod


class PolicyBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def setup_with_value(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __call__(self): raise NotImplementedError
    @abstractmethod
    def choose_action(self): raise NotImplementedError
    @abstractmethod
    def sample(self): raise NotImplementedError
    @abstractmethod
    def P(self): raise NotImplementedError
    @abstractmethod
    def logP(self): raise NotImplementedError
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
    def policy_network(self): raise NotImplementedError
    @property
    @abstractmethod
    def policy_optimizer(self): raise NotImplementedError
    @property
    @abstractmethod
    def can_pointwise_estimate(self): raise NotImplementedError
    @property
    @abstractmethod
    def can_density_estimate(self): raise NotImplementedError


class DiscretePolicyBase(PolicyBase):
    pass


class ContinuousPolicyBase(PolicyBase):
    pass


class GoalConditionedPolicyBase(PolicyBase):
    pass


class EpsilonGreedyPolicyBase(PolicyBase):
    pass


class GoalConditionedEpsilonGreedyPolicyBase(EpsilonGreedyPolicyBase, GoalConditionedPolicyBase):
    pass
