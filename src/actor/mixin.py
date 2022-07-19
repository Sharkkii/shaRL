#### Actor (Mixin Class) ####

from ..common import Component
from ..policy import EmptyPolicyBase
from ..policy import DiscretePolicy
from ..policy import ContinuousPolicy
from ..policy import GoalConditionedPolicy

from .base import EmptyActorBase
from .base import ActorBase
from .base import DiscreteControlActorBase
from .base import ContinuousControlActorBase
from .base import GoalConditionedActorBase


class EmptyActorMixin(EmptyActorBase):

    def __init__(self): return
    def setup(self): raise NotImplementedError
    def setup_with_critic(self): raise NotImplementedError
    def epochwise_preprocess(self): raise NotImplementedError
    def epochwise_postprocess(self): raise NotImplementedError
    def stepwise_preprocess(self): raise NotImplementedError
    def stepwise_postprocess(self): raise NotImplementedError
    def __call__(self): raise NotImplementedError
    def choose_action(self): raise NotImplementedError
    def update(self): raise NotImplementedError
    def update_policy(self): raise NotImplementedError
    def train(self): raise NotImplementedError
    def eval(self): raise NotImplementedError

    @property
    def interface(self): raise NotImplementedError
    @property
    def configuration(self): raise NotImplementedError
    @property
    def policy(self): raise NotImplementedError
    @property
    def can_pointwise_estimate(self): raise NotImplementedError
    @property
    def can_density_estimate(self): raise NotImplementedError


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
        use_default = True,
    ):
        if (policy is None):
            policy = DiscretePolicy(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )

        ActorMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            ActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True,
    ):
        if (interface is None):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (policy is None): return

        self._interface = interface
        self._configuration = configuration
        self._policy = policy
        self._become_available()

    def setup_with_critic(
        self,
        critic
    ):
        pass

    def epochwise_preprocess(
        self,
        epoch,
        n_epoch
    ):
        pass

    def epochwise_postprocess(
        self,
        epoch,
        n_epoch
    ):
        pass

    def stepwise_preprocess(
        self,
        step,
        n_step
    ):
        pass

    def stepwise_postprocess(
        self,
        step,
        n_step
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
        if (not isinstance(self.policy, EmptyPolicyBase)):
            self.policy.train()

    def eval(
        self
    ):
        if (not isinstance(self.policy, EmptyPolicyBase)):
            self.policy.eval()


class DiscreteControlActorMixin(ActorMixin, DiscreteControlActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        allow_setup = True,
        use_default = True,
    ):
        if (policy is None):
            policy = DiscretePolicy(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )
        
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            allow_setup = allow_setup,
            use_default = use_default,
        )
        if (allow_setup):
            DiscreteControlActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True,
    ):
        pass


class ContinuousControlActorMixin(ActorMixin, ContinuousControlActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        allow_setup = True,
        use_default = True,
    ):
        if (policy is None):
            policy = ContinuousPolicy(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )
        
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            allow_setup = allow_setup,
            use_default = use_default,
        )
        if (allow_setup):
            ContinuousControlActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True,
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
        SoftUpdateActorMixin.declare(self)
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
        use_default = True,
    ):
        if (policy is None):
            policy = GoalConditionedPolicy(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )
        
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            allow_setup = allow_setup,
            use_default = use_default,
        )
        if (allow_setup):
            GoalConditionedActorMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                policy = policy,
                use_default = use_default,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True,
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
