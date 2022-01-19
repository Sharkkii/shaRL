#### Helper ####

from ...const import SpaceType
from ...const import MeasureType
from ...const import cast_to_space_type
from ...network import get_default_measure_network
from ...optimizer import get_default_measure_optimizer
from ..policy import BasePolicy
from ..policy import Policy
from ..policy import DiscretePolicy
from ..policy import ContinuousPolicy
from ..policy import PseudoPolicy

def get_default_policy(
    env
):
    measure_type = MeasureType.POLICY
    default_policy_network = get_default_measure_network(
        env = env,
        measure_type = measure_type
    )
    default_policy_optimizer = get_default_measure_optimizer()

    if (cast_to_space_type(env) == SpaceType.DISCRETE):
        default_policy = DiscretePolicy(
            policy_network = default_policy_network,
            policy_optimizer = default_policy_optimizer
        )
    elif (cast_to_space_type(env) == SpaceType.CONTINUOUS):
        default_policy = ContinuousPolicy(
            policy_network = default_policy_network,
            policy_optimizer = default_policy_optimizer
        )
    else:
        default_policy = None
    
    return default_policy

def cast_to_policy(
    policy
):
    if (isinstance(policy, BasePolicy)):
        return policy
    else:
        return PseudoPolicy()
