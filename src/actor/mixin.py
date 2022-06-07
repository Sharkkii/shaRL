#### Actor (Mixin Class) ####

from ..common import Component
from ..policy import DiscretePolicy
from ..policy import ContinuousPolicy
from ..policy import GoalConditionedPolicy

from .base import ActorBase
from .base import DiscreteControlActorBase
from .base import ContinuousControlActorBase
from .base import GoalConditionedActorBase


class ActorMixin(ActorBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._policy = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def policy(self): return self._policy

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        allow_setup = True,
        use_default = False,
        default_policy = None
    ):
        if (default_policy is None):
            default_policy = DiscretePolicy

        ActorMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            ActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
                default_policy = default_policy
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = False,
        default_policy = None
    ):
        TPOLICY = default_policy

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if (policy is not None):
                raise ValueError("`policy` must be None if `use_default = True`")
            if (default_policy is None):
                raise ValueError("`default_policy` must not be None if `use_default = True`")

            policy = TPOLICY(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )

            # if (interface.tout is SpaceType.DISCRETE):
#                 policy = DiscretePolicy(
#                     interface = interface,
#                     use_default = True
#                 )
#             elif (interface.tout is SpaceType.CONTINUOUS):
#                 policy = ContinuousPolicy(
#                     interface = interface,
#                     use_default = True
#                 )
#             else:
#                 raise ValueError("invalid interface")

        else:
            if (policy is None):
                return

        self._interface = interface
        self._configuration = configuration
        self._policy = policy
        self._become_available()

    def setup_with_critic(
        self,
        critic
    ):
        pass

    @property
    def can_pointwise_estimate(
        self
    ):
        return (self.is_available and self.policy.can_pointwise_estimate)

    @property
    def can_density_estimate(
        self
    ):
        return (self.is_available and self.policy.can_density_estimate)

    def __call__(
        self,
        state,
        information = None
    ):
        return self.choose_action(
            state = state,
            information = information
        )

    def choose_action(
        self,
        state,
        information = None
    ):
        return self.policy.choose_action(
            state = state,
            information = information
        )

    def update(
        self,
        critic,
        history,
        n_step = 1
    ):
        for _ in range(n_step):
            self.update_policy(
                critic = critic,
                history = history
            )
    
    def update_policy(
        self,
        critic,
        history
    ):
        pass

    def train(
        self
    ):
        self.policy.train()

    def eval(
        self
    ):
        self.policy.eval()


class DiscreteControlActorMixin(ActorMixin, DiscreteControlActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        allow_setup = True,
        use_default = False,
        default_policy = None
    ):
        if (default_policy is None):
            default_policy = DiscretePolicy
        
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            allow_setup = allow_setup,
            use_default = use_default,
            default_policy = default_policy
        )
        if (allow_setup):
            DiscreteControlActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
                default_policy = default_policy
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = False,
        default_policy = None
    ):
        pass


class ContinuousControlActorMixin(ActorMixin, ContinuousControlActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        allow_setup = True,
        use_default = False,
        default_policy = None
    ):
        if (default_policy is None):
            default_policy = ContinuousPolicy
        
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            allow_setup = allow_setup,
            use_default = use_default,
            default_policy = default_policy
        )
        if (allow_setup):
            ContinuousControlActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
                default_policy = default_policy
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = False,
        default_policy = None
    ):
        pass


class SoftUpdateActorMixin(ActorBase):

    def declare(self):
        self._target_policy = None
        self._tau = None
    
    @property
    def target_policy(self): return self._target_policy
    @property
    def tau(self): return self._tau

    def __init__(
        self,
        tau = 0.01
    ):
        SoftUpdateActorMixin.declare()
        SoftUpdateActorMixin.setup(
            self,
            tau = tau
        )

    def setup(
        self,
        tau = 0.01
    ):
        self._target_policy = self.policy.copy()
        self._tau = tau

    def update(
        self,
        critic,
        history,
        n_step = 1
    ):
        for _ in range(n_step):
            self.update_policy(
                critic = critic,
                history = history
            )
            self.update_target_policy(
                critic = critic,
                history = history
            )

    def update_target_policy(
        self,
        critic,
        history
    ):
        pass
        # for theta, target_theta in zip(self.policy.policy_network.parameters(), self.target_policy.policy_network.parameters()):
        #     target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data


class GoalConditionedActorMixin(ActorMixin, GoalConditionedActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        allow_setup = True,
        use_default = False,
        default_policy = None
    ):
        if (default_policy is None):
            default_policy = GoalConditionedPolicy
        
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            allow_setup = allow_setup,
            use_default = use_default,
            default_policy = default_policy
        )
        if (allow_setup):
            GoalConditionedActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
                default_policy = default_policy
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = False,
        default_policy = None
    ):
        pass

    def __call__(
        self,
        state,
        goal,
        information = None
    ):
        return self.choose_action(
            state = state,
            goal = goal,
            information = information
        )

    def choose_action(
        self,
        state,
        goal,
        information = None
    ):
        return self.policy.choose_action(
            state = state,
            goal = goal,
            information = information
        )
