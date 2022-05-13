#### Helper ####

import gym
from ...common import EnvironmentInterface
from ...const import SpaceType
from ...const import MeasureType
from ...const import cast_to_space_type

def get_environment_interface(
        observation_space,
        action_space
    ):
        if (not (isinstance(observation_space, gym.spaces.Space) and isinstance(action_space, gym.spaces.Space))):
            return EnvironmentInterface(
                observation_type = SpaceType.NONE,
                action_type = SpaceType.NONE,
                observation_shape = None,
                action_shape = None,
                observation_ndim = None,
                action_ndim = None
            )
        
        if (type(observation_space) is gym.spaces.Discrete):
            observation_type = SpaceType.DISCRETE
        elif (type(observation_space) is gym.spaces.Box):
            observation_type = SpaceType.CONTINUOUS

        if (type(action_space) is gym.spaces.Discrete):
            action_type = SpaceType.DISCRETE
        elif (type(action_space) is gym.spaces.Box):
            action_type = SpaceType.CONTINUOUS
        
        observation_shape = observation_space.shape
        action_shape = action_space.shape
        observation_ndim = len(observation_shape)
        action_ndim = len(action_shape)

        return EnvironmentInterface(
            observation_type = observation_type,
            action_type = action_type,
            observation_shape = observation_shape,
            action_shape = action_shape,
            observation_ndim = observation_ndim,
            action_ndim = action_ndim
        )

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
