#### Policy ####

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..controller import Phases


class BasePolicy(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        self.policy_network = policy_network if callable(policy_network) else (lambda state: None)
        self.policy_optimizer = policy_optimizer

    @abstractmethod
    def __call__(
        self,
        state,
        action = None
    ):
        raise NotImplementedError

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        if callable(policy_network):
            self.policy_network = policy_network
        if (policy_optimizer is not None):
            self.policy_optimizer = policy_optimizer
            self.policy_optimizer.setup(
                network = self.policy_network
            )

    def train(
        self
    ):
        self.policy_network.train()
    
    def eval(
        self
    ):
        self.policy_network.eval()
    
    # @abstractmethod
    def P(
        self,
        **x
    ):
        return self.policy_network(**x)

    # @abstractmethod
    def logP(
        self,
        **x
    ):
        return torch.log(self.P(**x))

    @abstractmethod
    def sample(
        self,
        state,
        action_space,
        phase = Phases.NONE
    ):
        raise NotImplementedError

    # @abstractmethod
    def copy(
        self
    ):
        return copy.deepcopy(self)

class DiscretePolicy(BasePolicy):
    
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        
    def reset(
        self
    ):
        pass

    def setup(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
    
    def __call__(
        self,
        state
    ):
        return self.policy_network(state)
    
    def sample(
        self,
        state,
        action_space,
        phase = Phases.NONE
    ):
        return action_space.sample()

    def predict(
        self,
        state
    ):
        return self.policy_network.predict(state)

class ContinuousPolicy(BasePolicy):
    
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
    
    def reset(
        self
    ):
        pass

    def setup(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )

    def __call__(
        self,
        state,
        action
    ):
        return self.policy_network(state, action)
    
    def P(
        self,
        state,
        action
    ):
        return self.policy_network.P(state, action)
        
    def logP(
        self,
        state,
        action,
        eps = 1e-8
    ):
        return self.policy_network.logP(state, action, eps=eps)

    def sample(
        self,
        state,
        **kwargs
    ):
        return self.policy_network.sample(state, **kwargs)

    def predict(
        self,
        state,
        noise
    ):
        return self.policy_network.predict(state, noise)
        
Policy = DiscretePolicy

class QBasedPolicy(BasePolicy):
    
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None,
        copied_qvalue = None
    ):
        super().__init__(
            policy_network = None, # policy_network,
            policy_optimizer = None # policy_optimizer
        )
        # DiscreteQValue
        self.copied_qvalue = copied_qvalue

    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        super().setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )

    def __call__(
        self,
        state,
        action = None
    ):
        assert(action is None)
        q = self.copied_qvalue(state)
        p = F.softmax(q, dim=1)
        return p
    
    def sample(
        self,
        state,
        action_space,
        phase = Phases.NONE,
        eps = 0.05
    ):
        if (self.copied_qvalue is None):
            action = action_space.sample()
        else:
            # epsilon-greedy
            q = self.copied_qvalue(torch.from_numpy(state)).detach().numpy()
            action = np.argmax(q)
            r = np.random.rand()
            if (r <= action_space.n * eps):
                action = action_space.sample()
        return action
