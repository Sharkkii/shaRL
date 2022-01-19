#### Helper ####

from ...const import SpaceType
from ...const import MeasureType
from ...const import cast_to_space_type
from ...network import get_default_measure_network
from ...optimizer import get_default_measure_optimizer
from ..value import BaseValue
from ..value import BaseQValue
from ..value import Value
from ..value import DiscreteQValue
from ..value import ContinuousQValue
from ..value import PseudoValue
from ..value import PseudoQValue

def get_default_value(
    env
):
    measure_type = MeasureType.VALUE
    default_value_network = get_default_measure_network(
        env = env,
        measure_type = measure_type
    )
    default_value_optimizer = get_default_measure_optimizer()
    default_value = Value(
        value_network = default_value_network,
        value_optimizer = default_value_optimizer
    )
    return default_value

def get_default_qvalue(
    env
):
    measure_type = MeasureType.QVALUE
    default_qvalue_network = get_default_measure_network(
        env = env,
        measure_type = measure_type
    )
    default_qvalue_optimizer = get_default_measure_optimizer()

    if (cast_to_space_type(env) == SpaceType.DISCRETE):
        default_qvalue = DiscreteQValue(
            qvalue_network = default_qvalue_network,
            qvalue_optimizer = default_qvalue_optimizer
        )
    elif (cast_to_space_type(env) == SpaceType.CONTINUOUS):
        default_qvalue = ContinuousQValue(
            qvalue_network = default_qvalue_network,
            qvalue_optimizer = default_qvalue_optimizer
        )
    else:
        default_qvalue = None
    
    return default_qvalue

def cast_to_value(
    value
):
    if (isinstance(value, BaseValue)):
        return value
    else:
        return PseudoValue()

def cast_to_qvalue(
    qvalue
):
    if (isinstance(qvalue, BaseQValue)):
        return qvalue
    else:
        return PseudoQValue()
