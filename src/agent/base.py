#### Agent (Base Class) ####

from abc import ABC, abstractmethod


class AgentBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
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
    @abstractmethod
    def choose_action(self): raise NotImplementedError
    @abstractmethod
    def interact_with_env(self): raise NotImplementedError
    @abstractmethod
    def update_actor(self): raise NotImplementedError
    @abstractmethod
    def update_critic(self): raise NotImplementedError
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
    def actor(self): raise NotImplementedError
    @property
    @abstractmethod
    def critic(self): raise NotImplementedError
    # @property
    # @abstractmethod
    # def env(self): raise NotImplementedError
    # @property
    # @abstractmethod
    # def memory(self): raise NotImplementedError


class DiscreteControlAgentBase(AgentBase):
    pass


class ContinuousControlAgentBase(AgentBase):
    pass


class GoalConditionedAgentBase(AgentBase):
    pass