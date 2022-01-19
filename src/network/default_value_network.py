#### Default Value Network ####

from .value_network import ValueNetwork
from .value_network import QValueNetwork
from .value_network import DiscreteQValueNetwork
from .value_network import ContinuousQValueNetwork

DefaultValueNetwork = ValueNetwork
DefaultQValueNetwork = DiscreteQValueNetwork
DefaultDiscreteQValueNetwork = DiscreteQValueNetwork
DefaultContinuousQValueNetwork = ContinuousQValueNetwork
