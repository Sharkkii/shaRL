#### Helper ####

from ...const import SpaceType
from ...const import MeasureType
from ...const import cast_to_space_type


def get_compatible_interface(
    env,
    measure_type
):

    if (cast_to_space_type(env) == SpaceType.DISCRETE):

        d_observation = env.observation_space.shape[0]
        d_action = env.action_space.n

        if (measure_type == MeasureType.VALUE):
            return (d_observation, 1)
        elif (measure_type == MeasureType.QVALUE):
            return (d_observation, d_action)
        elif (measure_type == MeasureType.POLICY):
            return (d_observation, d_action)

    elif (cast_to_space_type(env) == SpaceType.CONTINUOUS):

        d_observation = env.observation_space.shape[0]
        d_action = env.action_space.shape[0]

        if (measure_type == MeasureType.VALUE):
            return (d_observation, 1)
        elif (measure_type == MeasureType.QVALUE):
            return (d_observation + d_action, 1)
        elif (measure_type == MeasureType.POLICY):
            return (d_observation + d_action, 1)

    return None
