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
        use_default = False
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
        use_default = False
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
        use_default = False
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
        use_default = False
    ):
        GoalConditionedActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            use_default = use_default
        )
