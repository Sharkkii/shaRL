#### Agent ####

from .base import AgentBase
from .mixin import AgentMixin
from .mixin import DiscreteControlAgentMixin
from .mixin import ContinuousControlAgentMixin
from .mixin import GoalConditionedAgentMixin


class Agent(AgentMixin, AgentBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False
    ):
        AgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = use_default
        )


class DiscreteControlAgent(DiscreteControlAgentMixin, AgentBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False
    ):
        DiscreteControlAgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = use_default
        )


class ContinuousControlAgent(ContinuousControlAgentMixin, AgentBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False
    ):
        ContinuousControlAgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = use_default
        )


class GoalConditionedAgent(GoalConditionedAgentMixin, AgentBase):
    
    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False
    ):
        GoalConditionedAgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = use_default
        )
