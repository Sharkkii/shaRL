#### Critic (Mixin Class) ####

from ..common import Component
from ..value import Value
from ..value import QValue
from ..value import Advantage
from ..value import DiscreteQValue
from ..value import ContinuousQValue
from ..value import DiscreteAdvantage
from ..value import ContinuousAdvantage

from .base import CriticBase
from .base import DiscreteControlCriticBase
from .base import ContinuousControlCriticBase


class CriticMixin(CriticBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._value = None
        self._qvalue = None
        self._advantage = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def value(self): return self._value
    @property
    def qvalue(self): return self._qvalue
    @property
    def advantage(self): return self._advantage

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        allow_setup = True,
        use_default = False,
        default_value = None,
        default_qvalue = None,
        default_advantage = None
    ):
        if (default_value is None):
            default_value = Value
        if (default_qvalue is None):
            default_qvalue = DiscreteQValue
        if (default_advantage is None):
            default_advantage = DiscreteAdvantage

        CriticMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            CriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                advantage = advantage,
                use_default = use_default,
                default_value = default_value,
                default_qvalue = default_qvalue,
                default_advantage = default_advantage
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        use_default = False,
        default_value = None,
        default_qvalue = None,
        default_advantage = None
    ):
        TVALUE = default_value
        TQVALUE = default_qvalue
        TADVANTAGE = default_advantage

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((value is not None) or (qvalue is not None) or (advantage is not None)):
                raise ValueError("`value` & `qvalue` & `advantage` must be None if `use_default = True`")
            if ((default_value is None) or (default_qvalue is None) or (default_advantage is None)):
                raise ValueError("`default_value` & `default_qvalue` & `default_advantage` must not be None if `use_default = True`")
            
            value = TVALUE(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )
            qvalue = TQVALUE(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )
            advantage = TADVANTAGE(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )

            # if (interface.tout is SpaceType.DISCRETE):
            #     qvalue = DiscreteQValue(
            #         interface = interface,
            #         use_default = True
            #     )
            # elif (interface.tout is SpaceType.CONTINUOUS):
            #     qvalue = ContinuousQValue(
            #         interface = interface,
            #         use_default = True
            #     )
            # else:
            #     raise ValueError("invalid interface")
        
        else:
            # for compatibility (TO BE FIXED)
            # if ((value is None) or (qvalue is None) or (advantage is None)):
            #     return

            if ((value is None) or (qvalue is None)):
                return
            elif (advantage is None):
                from ..const import SpaceType
                from ..common import AgentInterface
                advantage = TADVANTAGE(
                    interface = AgentInterface(
                        sin = (4,),
                        sout = (2,),
                        tin = SpaceType.CONTINUOUS,
                        tout = SpaceType.DISCRETE
                    ),
                    # configuration = configuration,
                    use_default = True
                )
            
        self._interface = interface
        self._configuration = configuration
        self._value = value
        self._qvalue = qvalue
        self._advantage = advantage
        self._become_available()

    def setup_with_actor(
        self,
        actor
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

    def update(
        self,
        actor,
        history,
        n_step = 1
    ):
        for _ in range(n_step):
            self.update_value(
                actor = actor,
                history = history
            )
            self.update_qvalue(
                actor = actor,
                history = history
            )
    
    def update_value(
        self,
        actor,
        history
    ):
        pass

    def update_qvalue(
        self,
        actor,
        history
    ):
        pass

    def update_advantage(
        self,
        actor,
        history
    ):
        pass

    def train(
        self
    ):
        self.value.train()
        self.qvalue.train()
        self.advantage.train()

    def eval(
        self
    ):
        self.value.eval()
        self.qvalue.eval()
        self.advantage.eval()


class DiscreteControlCriticMixin(CriticMixin, DiscreteControlCriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        allow_setup = True,
        use_default = False,
        default_value = None,
        default_qvalue = None,
        default_advantage = None
    ):
        if (default_value is None):
            default_value = Value
        if (default_qvalue is None):
            default_qvalue = DiscreteQValue
        if (default_advantage is None):
            default_advantage = DiscreteAdvantage

        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            advantage = advantage,
            allow_setup = allow_setup,
            use_default = use_default,
            default_value = default_value,
            default_qvalue = default_qvalue,
            default_advantage = default_advantage
        )
        if (allow_setup):
            DiscreteControlCriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                advantage = advantage,
                use_default = use_default,
                default_value = default_value,
                default_qvalue = default_qvalue,
                default_advantage = default_advantage
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        use_default = False,
        default_value = None,
        default_qvalue = None,
        default_advantage = None
    ):
        pass


class ContinuousControlCriticMixin(CriticMixin, ContinuousControlCriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        allow_setup = True,
        use_default = False,
        default_value = None,
        default_qvalue = None,
        default_advantage = None
    ):
        if (default_value is None):
            default_value = Value
        if (default_qvalue is None):
            default_qvalue = ContinuousQValue
        if (default_advantage is None):
            default_advantage = ContinuousAdvantage

        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            advantage = advantage,
            allow_setup = allow_setup,
            use_default = use_default,
            default_value = default_value,
            default_qvalue = default_qvalue,
            default_advantage = default_advantage
        )
        if (allow_setup):
            CriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                advantage = advantage,
                use_default = use_default,
                default_value = default_value,
                default_qvalue = default_qvalue,
                default_advantage = default_advantage
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        use_default = False,
        default_value = None,
        default_qvalue = None,
        default_advantage = None
    ):
        pass


class SoftUpdateCriticMixin(CriticBase):

    def declare(self):
        self._target_value = None
        self._target_qvalue = None
        self._target_advantage = None
        self._tau = None
    
    @property
    def target_value(self): return self._target_value
    @property
    def target_qvalue(self): return self._target_qvalue
    @property
    def target_advantage(self): return self._target_advantage
    @property
    def tau(self): return self._tau

    def __init__(
        self,
        tau = 0.01
    ):
        SoftUpdateCriticMixin.declare(self)
        SoftUpdateCriticMixin.setup(
            self,
            tau = tau
        )

    def setup(
        self,
        tau = 0.01
    ):
        self._target_value = self.value.copy()
        self._target_qvalue = self.qvalue.copy()
        self._target_advantage = self.advantage.copy()
        self._tau = tau

    def update(
        self,
        actor,
        history,
        n_step = 1
    ):
        for _ in range(n_step):
            self.update_value(
                actor = actor,
                history = history
            )
            self.update_qvalue(
                actor = actor,
                history = history
            )
            self.update_advantage(
                actor = actor,
                history = history
            )
            self.update_target_value(
                actor = actor,
                history = history
            )
            self.update_target_qvalue(
                actor = actor,
                history = history
            )
            self.update_target_advantage(
                actor = actor,
                history = history
            )

    def update_target_value(
        self,
        actor,
        history = None
    ):
        for theta, target_theta in zip(self.value.value_network.parameters(), self.target_value.value_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

    def update_target_qvalue(
        self,
        actor,
        history = None
    ):
        for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data
    
    def update_target_advantage(
        self,
        actor,
        history = None
    ):
        for theta, target_theta in zip(self.advantage.advantage_network.parameters(), self.target_advantage.advantage_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data
