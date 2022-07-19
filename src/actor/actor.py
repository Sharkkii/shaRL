#### Actor ####

from .base import ActorBase
from .mixin import EmptyActorMixin
from .mixin import ActorMixin
from .mixin import DiscreteControlActorMixin
from .mixin import ContinuousControlActorMixin
from .mixin import GoalConditionedActorMixin


class EmptyActor(EmptyActorMixin):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        return


class Actor(ActorMixin, ActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True
    ):
        ActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            use_default = use_default
        )


class DiscreteControlActor(DiscreteControlActorMixin, ActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True
    ):
        DiscreteControlActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            use_default = use_default
        )


class ContinuousControlActor(ContinuousControlActorMixin, ActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True
    ):
        ContinuousControlActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            use_default = use_default
        )


class GoalConditionedActor(GoalConditionedActorMixin, ActorBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = True
    ):
        GoalConditionedActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            use_default = use_default
        )
