#### Value (Mixin Class) ####

import copy
import torch
from ..common import Component
from ..network import ValueNetwork
from ..network import QValueNetwork
from ..network import DiscreteQValueNetwork
from ..network import ContinuousQValueNetwork
from ..network import AdvantageNetwork
from ..network import DiscreteAdvantageNetwork
from ..network import ContinuousAdvantageNetwork
from ..optimizer import MeasureOptimizer

from .base import ValueBase
from .base import QValueBase
from .base import DiscreteQValueBase
from .base import ContinuousQValueBase
from .base import AdvantageBase
from .base import DiscreteAdvantageBase
from .base import ContinuousAdvantageBase


class ValueMixin(ValueBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._value_network = None
        self._value_optimizer = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def value_network(self): return self._value_network
    @property
    def value_optimizer(self): return self._value_optimizer

    def __init__(
        self,
        interface = None,
        configuration = None,
        value_network = None,
        value_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_value_network = None,
        default_value_optimizer = None
    ):
        if (default_value_network is None):
            default_value_network = ValueNetwork
        if (default_value_optimizer is None):
            default_value_optimizer = MeasureOptimizer
        
        ValueMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            ValueMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                value_network = value_network,
                value_optimizer = value_optimizer,
                use_default = use_default,
                default_value_network = default_value_network,
                default_value_optimizer = default_value_optimizer,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        value_network = None,
        value_optimizer = None,
        use_default = False,
        default_value_network = None,
        default_value_optimizer = None
    ):
        TValueNetwork = default_value_network
        TValueOptimizer = default_value_optimizer

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((value_network is not None) or (value_optimizer is not None)):
                raise ValueError("`value_network` & `value_optimizer` must be None if `use_default = True`")
            if ((default_value_network is None) or (default_value_optimizer is None)):
                raise ValueError("`default_value` & `default_value_optimizer` must not be None if `use_default = True`")
            
            value_network = TValueNetwork(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )
            value_optimizer = TValueOptimizer(torch.optim.Adam, lr=1e-3)

        else:
            if ((value_network is None) or (value_optimizer is None)):
                return

        self._interface = interface
        self._configuration = configuration
        self._value_network = value_network
        self._value_optimizer = value_optimizer
        self.value_optimizer.setup(
            network = self.value_network
        )
        self._become_available()

    def __call__(
        self,
        state
    ):
        return self.value_network(state)

    def train(
        self
    ):
        self.value_network.train()
    
    def eval(
        self
    ):
        self.value_network.eval()

    def copy(
        self
    ):
        return copy.deepcopy(self)


class QValueMixin(QValueBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._qvalue_network = None
        self._qvalue_optimizer = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def qvalue_network(self): return self._qvalue_network
    @property
    def qvalue_optimizer(self): return self._qvalue_optimizer

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_qvalue_network = None,
        default_qvalue_optimizer = None
    ):
        if (default_qvalue_network is None):
            default_qvalue_network = QValueNetwork
        if (default_qvalue_optimizer is None):
            default_qvalue_optimizer = MeasureOptimizer
        
        QValueMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            QValueMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                qvalue_network = qvalue_network,
                qvalue_optimizer = qvalue_optimizer,
                use_default = use_default,
                default_qvalue_network = default_qvalue_network,
                default_qvalue_optimizer = default_qvalue_optimizer,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        use_default = False,
        default_qvalue_network = None,
        default_qvalue_optimizer = None
    ):
        TQValueNetwork = default_qvalue_network
        TQValueOptimizer = default_qvalue_optimizer

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((qvalue_network is not None) or (qvalue_optimizer is not None)):
                raise ValueError("`qvalue_network` & `qvalue_optimizer` must be None if `use_default = True`")
            if ((default_qvalue_network is None) or (default_qvalue_optimizer is None)):
                raise ValueError("`default_qvalue` & `default_qvalue_optimizer` must not be None if `use_default = True`")

            # if (interface.tout is SpaceType.DISCRETE):
            #     qvalue_network = DiscreteQValueNetwork(
            #         interface = interface,
            #         use_default = True
            #     )
            # elif (interface.tout is SpaceType.CONTINUOUS):
            #     qvalue_network = ContinuousQValueNetwork(
            #         interface = interface,
            #         use_default = True
            #     )
            # else:
            #     raise ValueError("invalid interface")
            
            qvalue_network = TQValueNetwork(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )
            qvalue_optimizer = TQValueOptimizer(torch.optim.Adam, lr=1e-3)

        else:
            if ((qvalue_network is None) or (qvalue_optimizer is None)):
                return

        self._interface = interface
        self._configuration = configuration
        self._qvalue_network = qvalue_network
        self._qvalue_optimizer = qvalue_optimizer
        self.qvalue_optimizer.setup(
            network = self.qvalue_network
        )
        self._become_available()

    def __call__(
        self,
        state,
        # action = None
    ):
        # assert(action is None)
        return self.qvalue_network(state)

    def train(
        self
    ):
        self.qvalue_network.train()
    
    def eval(
        self
    ):
        self.qvalue_network.eval()

    def copy(
        self
    ):
        return copy.deepcopy(self)


class DiscreteQValueMixin(QValueMixin, DiscreteQValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_qvalue_network = None,
        default_qvalue_optimizer = None
    ):
        if (default_qvalue_network is None):
            default_qvalue_network = DiscreteQValueNetwork
        if (default_qvalue_optimizer is None):
            default_qvalue_optimizer = MeasureOptimizer

        QValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_qvalue_network = default_qvalue_network,
            default_qvalue_optimizer = default_qvalue_optimizer
        )
        if (allow_setup):
            DiscreteQValueMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                qvalue_network = qvalue_network,
                qvalue_optimizer = qvalue_optimizer,
                use_default = use_default,
                default_qvalue_network = default_qvalue_network,
                default_qvalue_optimizer = default_qvalue_optimizer
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        use_default = False,
        default_qvalue_network = None,
        default_qvalue_optimizer = None
    ):
        pass


class ContinuousQValueMixin(QValueMixin, ContinuousQValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_qvalue_network = None,
        default_qvalue_optimizer = None
    ):
        if (default_qvalue_network is None):
            default_qvalue_network = ContinuousQValueNetwork
        if (default_qvalue_optimizer is None):
            default_qvalue_optimizer = MeasureOptimizer

        QValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_qvalue_network = default_qvalue_network,
            default_qvalue_optimizer = default_qvalue_optimizer
        )
        if (allow_setup):
            ContinuousQValueMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                qvalue_network = qvalue_network,
                qvalue_optimizer = qvalue_optimizer,
                use_default = use_default,
                default_qvalue_network = default_qvalue_network,
                default_qvalue_optimizer = default_qvalue_optimizer
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        use_default = False,
        default_qvalue_network = None,
        default_qvalue_optimizer = None
    ):
        pass


class AdvantageMixin(AdvantageBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._advantage_network = None
        self._advantage_optimizer = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def advantage_network(self): return self._advantage_network
    @property
    def advantage_optimizer(self): return self._advantage_optimizer

    def __init__(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_advantage_network = None,
        default_advantage_optimizer = None
    ):
        if (default_advantage_network is None):
            default_advantage_network = AdvantageNetwork
        if (default_advantage_optimizer is None):
            default_advantage_optimizer = MeasureOptimizer
        
        AdvantageMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            AdvantageMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                advantage_network = advantage_network,
                advantage_optimizer = advantage_optimizer,
                use_default = use_default,
                default_advantage_network = default_advantage_network,
                default_advantage_optimizer = default_advantage_optimizer,
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        use_default = False,
        default_advantage_network = None,
        default_advantage_optimizer = None
    ):
        TAdvantageNetwork = default_advantage_network
        TAdvantageOptimizer = default_advantage_optimizer

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((advantage_network is not None) or (advantage_optimizer is not None)):
                raise ValueError("`advantage_network` & `advantage_optimizer` must be None if `use_default = True`")
            if ((default_advantage_network is None) or (default_advantage_optimizer is None)):
                raise ValueError("`default_advantage` & `default_advantage_optimizer` must not be None if `use_default = True`")
            
            advantage_network = TAdvantageNetwork(
                interface = interface,
                # configuration = configuration,
                use_default = True
            )
            advantage_optimizer = TAdvantageOptimizer(torch.optim.Adam, lr=1e-3)

        else:
            if ((advantage_network is None) or (advantage_optimizer is None)):
                return

        self._interface = interface
        self._configuration = configuration
        self._advantage_network = advantage_network
        self._advantage_optimizer = advantage_optimizer
        self.advantage_optimizer.setup(
            network = self.advantage_network
        )
        self._become_available()

    def __call__(
        self,
        state,
        # action = None
    ):
        # assert(action is None)
        return self.advantage_network(state)

    def train(
        self
    ):
        self.advantage_network.train()
    
    def eval(
        self
    ):
        self.advantage_network.eval()

    def copy(
        self
    ):
        return copy.deepcopy(self)


class DiscreteAdvantageMixin(AdvantageMixin, DiscreteAdvantageBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_advantage_network = None,
        default_advantage_optimizer = None
    ):
        if (default_advantage_network is None):
            default_advantage_network = DiscreteAdvantageNetwork
        if (default_advantage_optimizer is None):
            default_advantage_optimizer = MeasureOptimizer

        AdvantageMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            advantage_network = advantage_network,
            advantage_optimizer = advantage_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_advantage_network = default_advantage_network,
            default_advantage_optimizer = default_advantage_optimizer
        )
        if (allow_setup):
            DiscreteAdvantageMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                advantage_network = advantage_network,
                advantage_optimizer = advantage_optimizer,
                use_default = use_default,
                default_advantage_network = default_advantage_network,
                default_advantage_optimizer = default_advantage_optimizer
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        use_default = False,
        default_advantage_network = None,
        default_advantage_optimizer = None
    ):
        pass


class ContinuousAdvantageMixin(AdvantageMixin, ContinuousAdvantageBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        allow_setup = True,
        use_default = False,
        default_advantage_network = None,
        default_advantage_optimizer = None
    ):
        if (default_advantage_network is None):
            default_advantage_network = ContinuousAdvantageNetwork
        if (default_advantage_optimizer is None):
            default_advantage_optimizer = MeasureOptimizer

        AdvantageMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            advantage_network = advantage_network,
            advantage_optimizer = advantage_optimizer,
            allow_setup = allow_setup,
            use_default = use_default,
            default_advantage_network = default_advantage_network,
            default_advantage_optimizer = default_advantage_optimizer
        )
        if (allow_setup):
            ContinuousAdvantageMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                advantage_network = advantage_network,
                advantage_optimizer = advantage_optimizer,
                use_default = use_default,
                default_advantage_network = default_advantage_network,
                default_advantage_optimizer = default_advantage_optimizer
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        use_default = False,
        default_advantage_network = None,
        default_advantage_optimizer = None
    ):
        pass
