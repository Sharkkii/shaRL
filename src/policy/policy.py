#### Policy ####

import warnings
from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..const import SpaceType
from ..const import PhaseType
from ..common import AgentInterface
from ..network import PolicyNetwork
from ..network import DiscretePolicyNetwork
from ..network import ContinuousPolicyNetwork
from ..network import cast_to_measure_network
from ..optimizer import Optimizer

class BasePolicy(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None,
        interface = None,
        use_default = False
    ):
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
            policy_optimizer = Optimizer(torch.optim.Adam)

        self.policy_network = None # cast_to_measure_network(policy_network)
        self.policy_optimizer = None # policy_optimizer
        self._is_available = False
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
        return self.policy_network(state)

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
        state,
        action = None
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

class QBasedPolicy(BasePolicy):
    
    def check_whether_available(f):
        def wrapper(self, *args, **kwargs):
            if (self.reference_qvalue is None):
                raise Exception(f"Call `QBasedPolicy.setup` before using `QBasedPolicy.{ f.__name__ }`")
            return f(self, *args, **kwargs)
        return wrapper

    def __init__(
        self,
        policy_network = None,
        policy_optimizer = None,
        reference_qvalue = None # read-only
    ):
        super().__init__(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
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
            self.setup_reference_qvalue(reference_qvalue)
    
    def setup_reference_qvalue(
        self,
        reference_qvalue
    ):
        assert(reference_qvalue is not None)
        self.reference_qvalue = reference_qvalue.copy()

    @check_whether_available
    def __call__(
        self,
        state,
        action = None
    ):
        assert(action is None)
        q = self.reference_qvalue(state)
        return q

    @check_whether_available
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
    
    @check_whether_available
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

    @check_whether_available
    def sample(
        self,
        state,
        action_space,
        phase,
        eps = 0.0
    ):
        # epsilon-greedy
        with torch.no_grad():

            state = torch.from_numpy(state)
            q = self.reference_qvalue(state).numpy()

            if (phase in [PhaseType.TRAINING]):
                r = np.random.rand()
                if (r <= action_space.n * eps):
                    action = action_space.sample()
                else:
                    action = np.argmax(q)

            elif (phase in [PhaseType.TEST]):
                action = np.argmax(q)

        action = np.int64(action)
        return action
