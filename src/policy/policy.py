#### Policy ####

import warnings
import numpy as np
import torch
import torch.nn.functional as F
import gym

from ..const import SpaceType
from ..const import PhaseType
from ..common import QValueReference

from .base import PolicyBase
from .mixin import PolicyMixin
from .mixin import DiscretePolicyMixin
from .mixin import ContinuousPolicyMixin
from .mixin import GoalConditionedPolicyMixin


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
    
    def choose_action(
        self,
        state,
        goal,
        information = None
    ):
        if (information is None):
            information = { "meta_policy": "greedy" }
        if ("meta_policy" not in information):
            raise ValueError("`information` must have the key 'meta_policy'.")
        meta_policy = information["meta_policy"]

        if (meta_policy == "greedy"):
            action = GoalConditionedPolicyMixin.choose_action(
                self,
                state = state,
                goal = goal,
                information = information
            )

        elif (meta_policy == "mixed"):
            r = np.random.rand()
            eps = 0.20
            if (r <= eps):
                action_space = information["action_space"]
                action = GoalConditionedPolicyMixin.sample(
                    self,
                    state = state,
                    goal = goal,
                    action_space = action_space
                    # information = information
                )
            else:
                action = GoalConditionedPolicyMixin.choose_action(
                    self,
                    state = state,
                    goal = goal,
                    information = information
                )

        elif (meta_policy == "random"):
            action_space = information["action_space"]
            action = GoalConditionedPolicyMixin.sample(
                self,
                state = state,
                goal = goal,
                action_space = action_space
                # information = information
            )
        else:
            raise ValueError()

        action = np.int64(action)
        return action


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
        self._qvalue_reference = None

    @property
    def qvalue_reference(self): return self._qvalue_reference

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        qvalue_reference = None, # read-only
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
                qvalue_reference = qvalue_reference,
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
        qvalue_reference = None,
        use_default = False,
        default_policy_network = None,
        default_policy_optimizer = None
    ):
        self.setup_with_value(
            qvalue = qvalue_reference
        )

    def setup_with_value(
        self,
        value = None,
        qvalue = None
    ):
        self._qvalue_reference = QValueReference(
            target = qvalue
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

            q = self.qvalue_reference(state).numpy()

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


class BasePolicy(): pass

class PseudoPolicy(BasePolicy): pass
