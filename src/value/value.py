#### Value function ####

from abc import ABC, ABCMeta, abstractmethod
import copy
import torch
import torch.nn as nn

from ..const import SpaceType
from ..common import AgentInterface
from ..common import Component
from ..network import ValueNetwork
from ..network import QValueNetwork
from ..network import DiscreteQValueNetwork
from ..network import ContinuousQValueNetwork
from ..optimizer import MeasureOptimizer


class ValueBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def setup_with_policy(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __call__(self): raise NotImplementedError
    # @abstractmethod
    # def update(self): raise NotImplementedError
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
    def value_network(self): raise NotImplementedError
    @property
    @abstractmethod
    def value_optimizer(self): raise NotImplementedError


class QValueBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def setup_with_policy(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __call__(self): raise NotImplementedError
    # @abstractmethod
    # def update(self): raise NotImplementedError
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
    def qvalue_network(self): raise NotImplementedError
    @property
    @abstractmethod
    def qvalue_optimizer(self): raise NotImplementedError


class DiscreteQValueBase(QValueBase):
    pass


class ContinuousQValueBase(QValueBase):
    pass


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


class Value(ValueMixin, ValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value_network = None,
        value_optimizer = None,
        use_default = False
    ):
        ValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value_network = value_network,
            value_optimizer = value_optimizer,
            use_default = use_default
        )


class QValue(QValueMixin, QValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        use_default = False
    ):
        QValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            use_default = use_default
        )


class DiscreteQValue(DiscreteQValueMixin, QValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        use_default = False
    ):
        DiscreteQValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            use_default = use_default
        )


class ContinuousQValue(ContinuousQValueMixin, QValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        use_default = False
    ):
        ContinuousQValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            use_default = use_default
        )


class BaseValue(ABC): pass

class PseudoValue(BaseValue): pass

class BaseQValue(ABC): pass

class PseudoQValue(BaseQValue): pass
