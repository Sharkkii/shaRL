#### Value ####

from .base import ValueBase
from .base import QValueBase
from .mixin import EmptyValueMixin
from .mixin import EmptyQValueMixin
from .mixin import ValueMixin
from .mixin import QValueMixin
from .mixin import DiscreteQValueMixin
from .mixin import ContinuousQValueMixin


class EmptyValue(EmptyValueMixin):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        return


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


class EmptyQValue(EmptyQValueMixin):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        return


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


class BaseValue(): pass

class PseudoValue(BaseValue): pass

class BaseQValue(): pass

class PseudoQValue(BaseQValue): pass
