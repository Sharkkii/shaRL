#### Critic ####

from .base import CriticBase
from .mixin import EmptyCriticMixin
from .mixin import CriticMixin
from .mixin import DiscreteControlCriticMixin
from .mixin import ContinuousControlCriticMixin


class EmptyCritic(EmptyCriticMixin):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        return


class Critic(CriticMixin, CriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False
    ):
        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            use_default = use_default
        )


class DiscreteControlCritic(DiscreteControlCriticMixin, CriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False
    ):
        DiscreteControlCriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            use_default = use_default
        )


class ContinuousControlCritic(ContinuousControlCriticMixin, CriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False
    ):
        ContinuousControlCriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            use_default = use_default
        )
