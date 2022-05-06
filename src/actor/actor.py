#### Actor ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from gym.spaces import Box, Discrete

from ..const import PhaseType
from ..policy import Policy
from ..policy import PseudoPolicy
from ..policy import cast_to_policy


class BaseActor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy = None,
        use_default = False
    ):
        if (use_default):
            if (policy is not None):
                raise ValueError("`policy` must be None if `use_default = True`")
            policy = Policy(use_default = True)

        self.policy = None # cast_to_policy(policy)
        self.target_policy = None # self.policy.copy()
        self._is_available = False
        self.setup(
            policy = policy
        )

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        env = None, # will be deprecated
        policy_network = None, # will be deprecated
        policy_optimizer = None, # will be deprecated
        policy = None
    ):
        if (policy is not None):
            self.policy = policy
            self.target_policy = self.policy.copy()
            self._become_available()
    
    @abstractmethod
    def setup_with_critic(
        self,
        critic
    ):
        raise NotImplementedError

    @abstractmethod
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        raise NotImplementedError

    @abstractmethod
    def setup_on_every_step(
        self,
        step,
        n_step
    ):
        raise NotImplementedError
    
    @property
    def is_available(
        self
    ):
        return self._is_available

    def _become_available(
        self
    ):
        self._is_available = True

    def _become_unavailable(
        self
    ):
        self._is_available = False
    
    def train(
        self
    ):
        self.policy.train()

    def eval(
        self
    ):
        self.policy.eval()

    def save(
        self,
        path_to_policy
    ):
        self.policy.save(path_to_policy)
    
    def load(
        self,
        path_to_policy
    ):
        self.policy.load(path_to_policy)

    @abstractmethod
    def choose_action(
        self,
        state
    ):
        raise NotImplementedError

    @abstractmethod
    def update_policy(
        self,
        trajectory = None
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_policy(
        self,
        trajectory
    ):
        raise NotImplementedError
    
    @abstractmethod
    def update(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError

class Actor(BaseActor):

    def __init__(
        self,
        policy = None,
        use_default = False
    ):
        super().__init__(
            policy = policy,
            use_default = use_default
        )
    
    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        env = None, # will be deprecated
        policy_network = None, # will be deprecated
        policy_optimizer = None, # will be deprecated
        policy = None
    ):
        super().setup(
            policy = policy
        )
        # self.env = env
        # self.policy.setup(
        #     policy_network = policy_network,
        #     policy_optimizer = policy_optimizer
        # )
        # self.target_policy = self.policy.copy()

    def setup_with_critic(
        self,
        critic
    ):
        pass
    
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        pass

    def setup_on_every_step(
        self,
        step,
        n_step
    ):
        pass
    
    def choose_action(
        self,
        state,
        action_space = None, # deprecated
        phase = PhaseType.NONE
    ):
        action_space = self.env.action_space
        action = self.policy(state)
        if (action is None):
            action = action_space.sample()
        else:
            action = torch.argmax(action) if (type(action_space) is Discrete) else action
        return action
    
    def update_policy(
        self,
        critic,
        trajectory
    ):
        pass

    def update_target_policy(
        self,
        critic,
        trajectory
    ):
        pass

    def update(
        self,
        critic,
        trajectory,
        n_times = 1
    ):
        for _ in range(n_times):
            self.update_policy(critic, trajectory)
            self.update_target_policy(critic, trajectory)
