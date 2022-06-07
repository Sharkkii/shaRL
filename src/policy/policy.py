#### Policy ####

import warnings
from abc import ABC, ABCMeta, abstractmethod
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
from ..optimizer import MeasureOptimizer


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


class PolicyMixin(PolicyBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._policy_network = None
        self._policy_optimizer = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def policy_network(self): return self._policy_network
    @property
    def policy_optimizer(self): return self._policy_optimizer

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        if (default_policy_network is None):
            default_policy_network = PolicyNetwork
        if (default_policy_optimizer is None):
            default_policy_optimizer = MeasureOptimizer
        
        PolicyMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            PolicyMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                use_default = use_default,
                default_policy_network = default_policy_network,
                default_policy_optimizer = default_policy_optimizer,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        TPolicyNetwork = default_policy_network
        TPolicyOptimizer = default_policy_optimizer

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((policy_network is not None) or (policy_optimizer is not None)):
                raise ValueError("`policy_network` & `policy_optimizer` must be None if `use_default = True`")
            if ((default_policy_network is None) or (default_policy_optimizer is None)):
                raise ValueError("`default_policy` & `default_policy_optimizer` must not be None if `use_default = True`")

            # if (interface.tout is SpaceType.DISCRETE):
            #     policy_network = DiscretePolicyNetwork(
            #         interface = interface,
            #         use_default = True
            #     )
            # elif (interface.tout is SpaceType.CONTINUOUS):
            #     policy_network = ContinuousPolicyNetwork(
            #         interface = interface,
            #         use_default = True
            #     )
            # else:
            #     raise ValueError("invalid interface")
            
            policy_network = TPolicyNetwork(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )
            policy_optimizer = TPolicyOptimizer(torch.optim.Adam, lr=1e-3)

        else:
            if ((policy_network is None) or (policy_optimizer is None)):
                return

        self._interface = interface
        self._configuration = configuration
        self._policy_network = policy_network
        self._policy_optimizer = policy_optimizer
        self._become_available()

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

    def sample(
        self,
        state,
        action_space,
        # phase = PhaseType.NONE
    ):
        return action_space.sample()

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

    def train(
        self
    ):
        self.policy_network.train()
    
    def eval(
        self
    ):
        self.policy_network.eval()

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

    def copy(
        self
    ):
        return copy.deepcopy(self)


class DiscretePolicyMixin(PolicyMixin, DiscretePolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        if (default_policy_network is None):
            default_policy_network = DiscretePolicyNetwork
        if (default_policy_optimizer is None):
            default_policy_optimizer = MeasureOptimizer

        PolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_policy_network = default_policy_network,
            default_policy_optimizer = default_policy_optimizer
        )
        if (allow_setup):
            DiscretePolicyMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                use_default = use_default,
                default_policy_network = default_policy_network,
                default_policy_optimizer = default_policy_optimizer
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        pass

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


class ContinuousPolicyMixin(PolicyMixin, ContinuousPolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        if (default_policy_network is None):
            default_policy_network = ContinuousPolicyNetwork
        if (default_policy_optimizer is None):
            default_policy_optimizer = MeasureOptimizer

        PolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_policy_network = default_policy_network,
            default_policy_optimizer = default_policy_optimizer
        )
        if (allow_setup):
            ContinuousPolicyMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                use_default = use_default,
                default_policy_network = default_policy_network,
                default_policy_optimizer = default_policy_optimizer
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        pass

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


class GoalConditionedPolicyMixin(PolicyMixin, GoalConditionedPolicyBase):

    def __call__(
        self,
        state,
        goal
    ):
        return self.choose_action(
            state = state,
            goal = goal
        )

    def choose_action(
        self,
        state,
        goal,
        information = None
    ):
        x = torch.cat([state, goal])
        action = self.policy_network(x)
        action = torch.argmax(action)
        return action

    def P(
        self,
        state,
        goal,
        action = None
    ):
        x = torch.cat([state, goal], dim=1)
        return self.policy_network.P(x, action)

    def logP(
        self,
        state,
        goal,
        action = None
    ):
        x = torch.cat([state, goal], dim=1)
        return self.policy_network.logP(x, action)
    
    def sample(
        self,
        state,
        goal,
        action_space,
        # phase = PhaseType.NONE
    ):
        return action_space.sample()


class Policy(PolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        PolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class DiscretePolicy(DiscretePolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        DiscretePolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class ContinuousPolicy(ContinuousPolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        ContinuousPolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class GoalConditionedPolicy(GoalConditionedPolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        PolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class GoalConditionedDiscretePolicy(GoalConditionedPolicyMixin, DiscretePolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        DiscretePolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class GoalConditionedContinuousPolicy(GoalConditionedPolicyMixin, ContinuousPolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        ContinuousPolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class QValueBasedPolicy(DiscretePolicyMixin, PolicyBase):

    def declare(self):
        self._reference_qvalue = None

    @property
    def reference_qvalue(self): return self.reference_qvalue

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        reference_qvalue = None, # read-only
        allow_setup = True,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        DiscretePolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_policy_network = default_policy_network,
            default_policy_optimizer = default_policy_optimizer
        )
        self.declare()

        if (allow_setup):
            QValueBasedPolicy.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                reference_qvalue = reference_qvalue,
                use_default = use_default,
                default_policy_network = default_policy_network,
                default_policy_optimizer = default_policy_optimizer
            )
    
    def setup(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        reference_qvalue = None,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        if (reference_qvalue is None):
            raise ValueError("`reference_qvalue` must be 'QValue'.")
        self._reference_qvalue = reference_qvalue.copy()

    def __call__(
        self,
        state
    ):
        return self.choose_action(
            state = state
        )

    @Component.check_whether_available
    def choose_action(
        self,
        state,
        information = None,
        # random = False # will be in `information`
    ):
        if (type(information) is not dict):
            raise ValueError("`information` must be 'Dictionary'.")

        # action_space
        if ("action_space" not in information):
            raise ValueError("`information` must have 'action_space' key.")
        if (not isinstance(information["action_space"], gym.Space)):
            raise ValueError("`action_space` must be 'gym.spaces'.")
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

    def sample(
        self,
        state,
        information = None
    ):
        return self.choose_action(
            state,
            information = information
        )

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


class BasePolicy(ABC): pass

class PseudoPolicy(BasePolicy): pass
