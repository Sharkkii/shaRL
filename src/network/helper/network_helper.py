### Helper ####

from ...const import SpaceType
from ...const import MeasureType
from ...const import cast_to_space_type
from ...environment import get_compatible_interface
from ...network import DefaultNetwork
from ...network import BaseMeasureNetwork
from ...network import PseudoMeasureNetwork
from ...network import DefaultValueNetwork
from ...network import DefaultDiscreteQValueNetwork
from ...network import DefaultContinuousQValueNetwork
from ...network import DefaultDiscretePolicyNetwork
from ...network import DefaultContinuousPolicyNetwork

def get_default_network(
    env,
    measure_type
):
    d_in, d_out = get_compatible_interface(
        env = env,
        measure_type = measure_type
    )
    default_network = DefaultNetwork(
        d_in = d_in,
        d_out = d_out
    )
    return default_network

def get_default_measure_network(
    env,
    measure_type
):
    default_network = get_default_network(
        env = env,
        measure_type = measure_type
    )
    
    if (measure_type == MeasureType.VALUE):
        default_measure_network = DefaultValueNetwork(
            default_network
        )
    
    elif (measure_type == MeasureType.QVALUE):
        if (cast_to_space_type(env) == SpaceType.DISCRETE):
            default_measure_network = DefaultDiscreteQValueNetwork(
                default_network
            )
        elif (cast_to_space_type(env) == SpaceType.CONTINUOUS):
            default_measure_network = DefaultContinuousQValueNetwork(
                default_network
            )
    
    elif (measure_type == MeasureType.POLICY):
        if (cast_to_space_type(env) == SpaceType.DISCRETE):
            default_measure_network = DefaultDiscretePolicyNetwork(
                default_network
            )
        elif (cast_to_space_type(env) == SpaceType.CONTINUOUS):
            default_measure_network = DefaultContinuousPolicyNetwork(
                default_network
            )

    else:
        default_measure_network = None

    return default_measure_network

def cast_to_measure_network(
    measure_network
):
    if (isinstance(measure_network, BaseMeasureNetwork)):
        return measure_network
    else:
        return PseudoMeasureNetwork()
