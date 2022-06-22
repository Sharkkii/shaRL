#### Value ####

from .base import ValueBase
from .base import QValueBase
from .base import AdvantageBase
from .mixin import ValueMixin
from .mixin import QValueMixin
from .mixin import DiscreteQValueMixin
from .mixin import ContinuousQValueMixin
from .mixin import DuelingNetworkQValueMixin
from .mixin import DiscreteDuelingNetworkQValueMixin
from .mixin import ContinuousDuelingNetworkQValueMixin
from .mixin import AdvantageMixin
from .mixin import DiscreteAdvantageMixin
from .mixin import ContinuousAdvantageMixin


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


class DuelingNetworkQValue(DuelingNetworkQValueMixin, QValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        value_reference = None,
        advantage_reference = None,
        use_default = False
    ):
        DuelingNetworkQValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            value_reference = value_reference,
            advantage_reference = advantage_reference,
            use_default = use_default
        )


class DiscreteDuelingNetworkQValue(DiscreteDuelingNetworkQValueMixin, QValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        value_reference = None,
        advantage_reference = None,
        use_default = False
    ):
        DiscreteDuelingNetworkQValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            value_reference = value_reference,
            advantage_reference = advantage_reference,
            use_default = use_default
        )


class ContinuousDuelingNetworkQValue(ContinuousDuelingNetworkQValueMixin, QValueBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        qvalue_network = None,
        qvalue_optimizer = None,
        value_reference = None,
        advantage_reference = None,
        use_default = False
    ):
        ContinuousDuelingNetworkQValueMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer,
            value_reference = value_reference,
            advantage_reference = advantage_reference,
            use_default = use_default
        )

class Advantage(AdvantageMixin, AdvantageBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        use_default = False
    ):
        AdvantageMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            advantage_network = advantage_network,
            advantage_optimizer = advantage_optimizer,
            use_default = use_default
        )


class DiscreteAdvantage(DiscreteAdvantageMixin, AdvantageBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        use_default = False
    ):
        DiscreteAdvantageMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            advantage_network = advantage_network,
            advantage_optimizer = advantage_optimizer,
            use_default = use_default
        )


class ContinuousAdvantage(ContinuousAdvantageMixin, AdvantageBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        advantage_network = None,
        advantage_optimizer = None,
        use_default = False
    ):
        ContinuousAdvantageMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            advantage_network = advantage_network,
            advantage_optimizer = advantage_optimizer,
            use_default = use_default
        )


class BaseValue(): pass

class PseudoValue(BaseValue): pass

class BaseQValue(): pass

class PseudoQValue(BaseQValue): pass
