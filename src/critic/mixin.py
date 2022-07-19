#### Critic (Mixin Class) ####

from ..common import Component
from ..value import EmptyValueBase
from ..value import EmptyQValueBase
from ..value import Value
from ..value import DiscreteQValue
from ..value import ContinuousQValue

from .base import EmptyCriticBase
from .base import CriticBase
from .base import DiscreteControlCriticBase
from .base import ContinuousControlCriticBase


class EmptyCriticMixin(EmptyCriticBase):

    def __init__(self): return
    def setup(self): raise NotImplementedError
    def setup_with_actor(self): raise NotImplementedError
    def epochwise_preprocess(self): raise NotImplementedError
    def epochwise_postprocess(self): raise NotImplementedError
    def stepwise_preprocess(self): raise NotImplementedError
    def stepwise_postprocess(self): raise NotImplementedError
    def update(self): raise NotImplementedError
    def update_value(self): raise NotImplementedError
    def update_qvalue(self): raise NotImplementedError
    def train(self): raise NotImplementedError
    def eval(self): raise NotImplementedError

    @property
    def interface(self): raise NotImplementedError
    @property
    def configuration(self): raise NotImplementedError
    @property
    def value(self): raise NotImplementedError
    @property
    def qvalue(self): raise NotImplementedError


class CriticMixin(CriticBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._value = None
        self._qvalue = None

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
        use_default = True,
    ):
        if (value is None):
            value = Value(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )
        if (qvalue is None):
            qvalue = DiscreteQValue(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )

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
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = True,
    ):
        if (interface is None):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if ((value is None) or (qvalue is None)): return

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

    def train(
        self
    ):
        if (not isinstance(self.value, EmptyValueBase)):
            self.value.train()
        if (not isinstance(self.qvalue, EmptyQValueBase)):
            self.qvalue.train()

    def eval(
        self
    ):
        if (not isinstance(self.value, EmptyValueBase)):
            self.value.eval()
        if (not isinstance(self.qvalue, EmptyQValueBase)):
            self.qvalue.eval()


class DiscreteControlCriticMixin(CriticMixin, DiscreteControlCriticBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        allow_setup = True,
        use_default = True,
    ):
        if (value is None):
            value = Value(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )
        if (qvalue is None):
            qvalue = DiscreteQValue(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )

        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            allow_setup = allow_setup,
            use_default = use_default,
        )
        if (allow_setup):
            DiscreteControlCriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                use_default = use_default,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = True,
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
        use_default = True,
    ):
        if (value is None):
            value = Value(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )
        if (qvalue is None):
            qvalue = ContinuousQValue(
                interface = interface,
                configuration = configuration,
                use_default = use_default
            )

        CriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            allow_setup = allow_setup,
            use_default = use_default,
        )
        if (allow_setup):
            CriticMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value = value,
                qvalue = qvalue,
                use_default = use_default,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = True,
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
        SoftUpdateCriticMixin.declare(self)
        SoftUpdateCriticMixin.setup(
            self,
            tau = tau
        )

    def setup(
        self,
        tau = 0.01
    ):
        if (not isinstance(self.value, EmptyValueBase)):
            self._target_value = self.value.copy()
        if (not isinstance(self.qvalue, EmptyQValueBase)):
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
        actor,
        history = None
    ):
        if (self.target_value is None):
            return
        for theta, target_theta in zip(self.value.value_network.parameters(), self.target_value.value_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

    def update_target_qvalue(
        self,
        actor,
        history = None
    ):
        if (self.target_qvalue is None):
            return
        for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data
