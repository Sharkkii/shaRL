#### Policy ####

import warnings
from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

from ..const import SpaceType
from ..const import PhaseType
from ..common import AgentInterface
from ..common import Component
from ..network import PolicyNetwork
from ..network import DiscretePolicyNetwork
from ..network import ContinuousPolicyNetwork
from ..optimizer import Optimizer

class BasePolicy(Component, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None,
        interface = None,
        use_default = False
    ):
        Component.__init__(self)
        if (use_default):
            if (not ((policy_network is None) and (policy_optimizer is None))):
                raise ValueError("`policy_network` & `policy_optimizer` must be None if `use_default = True`")
            if (type(interface) is not AgentInterface):
                raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`")
            if (interface.tout is SpaceType.DISCRETE):
                policy_network = DiscretePolicyNetwork(
                    interface = interface,
                    use_default = True
                )
            elif (interface.tout is SpaceType.CONTINUOUS):
                policy_network = ContinuousPolicyNetwork(
                    interface = interface,
                    use_default = True
                )
            else:
                raise ValueError("invalid interface")
            policy_optimizer = Optimizer(torch.optim.Adam, lr=1e-3)

        self.policy_network = None
        self.policy_optimizer = None
        self.setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )

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
        if ((policy_network is not None) and (policy_optimizer is not None)):
            self.policy_network = policy_network
            self.policy_optimizer = policy_optimizer
            self.policy_optimizer.setup(
                network = self.policy_network
            )
            self._become_available()
            print(f"Policy.setup: { self.policy_network } & { self.policy_optimizer }")

    @property
    def can_pointwise_estimate(
        self
    ):
        return False

    @property
    def can_density_estimate(
        self
    ):
        return False

    def train(
        self
    ):
        self.policy_network.train()
    
    def eval(
        self
    ):
        self.policy_network.eval()

    def save(
        self,
        path_to_policy_network
    ):
        self.policy_network.save(path_to_policy_network)

    def load(
        self,
        path_to_policy_network
    ):
        self.policy_network.load(path_to_policy_network)
    
    @abstractmethod
    def P(
        self,
        state,
        action
    ):
        raise NotImplementedError

    @abstractmethod
    def logP(
        self,
        state,
        action
    ):
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        state,
        action_space,
        phase = PhaseType.NONE
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
        policy_optimizer = None,
        interface = None,
        use_default = False
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            interface = interface,
            use_default = use_default
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
        return self.choose_action(
            state = state
        )

    def choose_action(
        self,
        state,
        information = None
    ):
        action = self.policy_network(state)
        action = torch.argmax(action)
        return action

    def P(
        self,
        state,
        action = None
    ):
        return self.policy_network.P(state, action)

    def logP(
        self,
        state,
        action = None
    ):
        return self.policy_network.logP(state, action)
    
    def sample(
        self,
        state,
        action_space,
        phase = PhaseType.NONE
    ):
        return action_space.sample()

    def predict(
        self,
        state
    ):
        warnings.warn("`Policy.predict` is deprecated. Use `Policy.P` instead.")
        return self.policy_network.predict(state)

class ContinuousPolicy(BasePolicy):
    
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None,
        interface = None,
        use_default = False
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            interface = interface,
            use_default = use_default
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
        return self.choose_action(
            state = state
        )

    def choose_action(
        self,
        state,
        information = None
    ):
        action = self.policy_network(state)
        return action
    
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

class PseudoPolicy(BasePolicy):

    @classmethod
    def __raise_exception(cls):
        raise Exception("`PseudoPolicy` cannot be used as a policy.")

    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        pass

    def reset(
        self
    ):
        pass

    def setup(
        self,
        policy_network = None,
        policy_optimizer = None
    ):
        pass
    
    def train(
        self
    ):
        pass

    def eval(
        self
    ):
        pass

    def __call__(
        self,
        state,
        action = None
    ):
        PseudoPolicy.__raise_exception()

    def P(
        self,
        state,
        action = None
    ):
        PseudoPolicy.__raise_exception()

    def logP(
        self,
        state,
        action = None
    ):
        PseudoPolicy.__raise_exception()
    
    def sample(
        self,
        state,
        action_space,
        phase = PhaseType.NONE
    ):
        PseudoPolicy.__raise_exception()

    def save(
        self,
        path_to_policy_network
    ):
        PseudoPolicy.__raise_exception()

    def load(
        self,
        path_to_policy_network
    ):
        PseudoPolicy.__raise_exception()

class QValueBasedPolicy(BasePolicy):

    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None,
        reference_qvalue = None, # read-only
        interface = None,
        use_default = False
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            interface = interface,
            use_default = use_default
        )   
        if (reference_qvalue is not None):
            self.setup_reference_qvalue(reference_qvalue)

    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        policy_network = None,
        policy_optimizer = None,
        reference_qvalue = None
    ):
        super().setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        if (reference_qvalue is not None):
            # self.policy_network = policy_network
            # self.policy_optimizer = policy_optimizer
            # self.policy_optimizer.setup(
            #     network = self.policy_network
            # )
            self.setup_reference_qvalue(reference_qvalue)
            self._become_available()
            # print(f"Policy.setup: { self.policy_network } & { self.policy_optimizer }")

        # super().setup(
        #     policy_network = policy_network,
        #     policy_optimizer = policy_optimizer
        # )
    
    def setup_reference_qvalue(
        self,
        reference_qvalue
    ):
        if (reference_qvalue is None):
            raise ValueError("`reference_qvalue` must not be None.")
        self.reference_qvalue = reference_qvalue.copy()

    @Component.check_whether_available
    def __call__(
        self,
        state
    ):
        return self.choose_action(
            state = state
        )

    @Component.check_whether_available
    def P(
        self,
        state,
        action = None
    ):
        # softmax policy
        dim = state.ndim - 1
        q = self(state)
        p = F.softmax(q, dim=dim)
        if (action is None):
            return p
        else:
            return p[action]
    
    @Component.check_whether_available
    def logP(
        self,
        state,
        action = None
    ):
        # softmax policy
        dim = state.ndim - 1
        q = self(state)
        log_p = F.log_softmax(q, dim=dim)
        if (action is None):
            return log_p
        else:
            return log_p[action]

    @Component.check_whether_available
    def choose_action(
        self,
        state,
        information = None,
        # random = False # will be in `information`
    ):
        if (type(information) is not dict):
            raise ValueError("`information` must be a 'dictionary' object.")

        # action_space
        if ("action_space" not in information):
            raise ValueError("`information` must have 'action_space' key.")
        if (not isinstance(information["action_space"], gym.Space)):
            raise ValueError("`action_space` must be a 'gym.spaces' object.")
        action_space = information["action_space"]

        # phase
        if ("phase" not in information):
            raise ValueError("`information` must have 'phase' key.")
        if (type(information["phase"]) is not PhaseType):
            raise ValueError("`phase` must be 'PhaseType'.")
        phase = information["phase"]

        # eps
        if ("eps" not in information):
            raise ValueError("`information` must have 'eps' key.")
        if (type(information["eps"]) is not float):
            raise ValueError("`eps` must be 'float'.")
        eps = information["eps"]

        # epsilon-greedy
        with torch.no_grad():

            q = self.reference_qvalue(state).numpy()

            if (phase in [PhaseType.TEST]):
                action = np.argmax(q)
            
            else:
                r = np.random.rand()
                if (r <= action_space.n * eps):
                    action = action_space.sample()
                else:
                    action = np.argmax(q)

        action = np.int64(action)
        return action

    @Component.check_whether_available
    def sample(
        self,
        state,
        information = None
    ):
        return self.choose_action(
            state,
            information = information
        )
