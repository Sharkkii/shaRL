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
from .mixin import EpsilonGreedyPolicyMixin
from .mixin import GoalConditionedPolicyMixin
from .mixin import GoalConditionedEpsilonGreedyPolicyMixin


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


class EpsilonGreedyPolicy(EpsilonGreedyPolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        EpsilonGreedyPolicyMixin.__init__(
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
        information = None
    ):
        return EpsilonGreedyPolicy.choose_action(
            self,
            state = state,
            information = information
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


class GoalConditionedEpsilonGreedyPolicy(GoalConditionedEpsilonGreedyPolicyMixin, PolicyBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy_network = None,
        policy_optimizer = None,
        use_default = False
    ):
        GoalConditionedEpsilonGreedyPolicyMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer,
            use_default = use_default
        )


class QValueBasedEpsilonGreedyPolicy(EpsilonGreedyPolicyMixin, PolicyBase):

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
        EpsilonGreedyPolicyMixin.__init__(
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
            QValueBasedEpsilonGreedyPolicy.setup(
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

    @EpsilonGreedyPolicy.choose_action_wrapper
    def choose_action(
        self,
        state,
        information = None
    ):
        with torch.no_grad():
            q = self.qvalue_reference(state).numpy()
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
