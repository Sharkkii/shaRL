#### Critic ####

from abc import ABC, ABCMeta, abstractmethod

from ..const import SpaceType
from ..common import Component
from ..common import AgentInterface
from ..value import Value
from ..value import QValue
from ..value import DiscreteQValue
from ..value import ContinuousQValue


class CriticBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    @abstractmethod
    def setup_with_actor(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    # @abstractmethod
    # def __call__(self): raise NotImplementedError
    @abstractmethod
    def update(self): raise NotImplementedError
    @abstractmethod
    def update_value(self): raise NotImplementedError
    @abstractmethod
    def update_qvalue(self): raise NotImplementedError
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
    def value(self): raise NotImplementedError
    @property
    @abstractmethod
    def qvalue(self): raise NotImplementedError


class DiscreteControlCriticBase(CriticBase):
    pass


class ContinuousControlCriticBase(CriticBase):
    pass


class CriticMixin(CriticBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._policy = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def value(self): return self._value
    @property
    def qvalue(self): return self._qvalue

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        allow_setup = True,
        use_default = False,
        default_value = None,
        default_qvalue = None
    ):
        if (default_value is None):
            default_value = Value
        if (default_qvalue is None):
            default_qvalue = DiscreteQValue

        CriticMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            CriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                use_default = use_default,
                default_value = default_value,
                default_qvalue = default_qvalue
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False,
        default_value = None,
        default_qvalue = None
    ):
        TVALUE = default_value
        TQVALUE = default_qvalue

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((value is not None) or (qvalue is not None)):
                raise ValueError("`value` & `qvalue` must be None if `use_default = True`")
            if ((default_value is None) or (default_qvalue is None)):
                raise ValueError("`default_value` & `default_qvalue` must not be None if `use_default = True`")
            
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
            if ((value is None) or (qvalue is None)):
                return

        self._interface = interface
        self._configuration = configuration
        self._value = value
        self._qvalue = qvalue
        self._become_available()

    def setup_with_actor(
        self,
        actor
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

    def train(
        self
    ):
        self.value.train()
        self.qvalue.train()

    def eval(
        self
    ):
        self.value.eval()
        self.qvalue.eval()


class DiscreteControlCriticMixin(CriticMixin, DiscreteControlCriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        allow_setup = True,
        use_default = False,
        default_value = None,
        default_qvalue = None
    ):
        if (default_value is None):
            default_value = Value
        if (default_qvalue is None):
            default_qvalue = DiscreteQValue

        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            allow_setup = allow_setup,
            use_default = use_default,
            default_value = default_value,
            default_qvalue = default_qvalue
        )
        if (allow_setup):
            DiscreteControlCriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                use_default = use_default,
                default_value = default_value,
                default_qvalue = default_qvalue
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False,
        default_value = None,
        default_qvalue = None
    ):
        pass


class ContinuousControlCriticMixin(CriticMixin, ContinuousControlCriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        allow_setup = True,
        use_default = False,
        default_value = None,
        default_qvalue = None
    ):
        if (default_value is None):
            default_value = Value
        if (default_qvalue is None):
            default_qvalue = ContinuousQValue

        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            allow_setup = allow_setup,
            use_default = use_default,
            default_value = default_value,
            default_qvalue = default_qvalue
        )
        if (allow_setup):
            CriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                use_default = use_default,
                default_value = default_value,
                default_qvalue = default_qvalue
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False,
        default_value = None,
        default_qvalue = None
    ):
        pass


class SoftUpdateCriticMixin(CriticBase):

    def declare(self):
        self._target_value = None
        self._target_qvalue = None
        self._tau = None
    
    @property
    def target_value(self): return self._target_value
    @property
    def target_qvalue(self): return self._target_qvalue
    @property
    def tau(self): return self._tau

    def __init__(
        self,
        tau = 0.01
    ):
        SoftUpdateCriticMixin.declare()
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
            self.update_target_value(
                actor = actor,
                history = history
            )
            self.update_target_qvalue(
                actor = actor,
                history = history
            )

    def update_target_value(
        self,
        critic,
        history
    ):
        pass
        # for theta, target_theta in zip(self.value.value_network.parameters(), self.target_value.value_network.parameters()):
        #     target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

    def update_target_qvalue(
        self,
        critic,
        history
    ):
        pass
        # for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
        #     target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data


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
