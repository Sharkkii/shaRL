from enum import Enum
import gym

class Spaces(Enum):
    NONE = 0
    DISCRETE = 1
    CONTINUOUS = 2

def cast_to_spaces(
    env
):
    if (type(env.action_space) is gym.spaces.Discrete):
        return Spaces.DISCRETE
    elif (type(env.action_space) is gym.spaces.Box):
        return Spaces.CONTINUOUS
    else:
        return Spaces.NONE

def get_compatible_interface(
    env
):
    interface = {}
    is_discrete = (cast_to_spaces(env) == Spaces.DISCRETE)
    is_continuous = (cast_to_spaces(env) == Spaces.CONTINUOUS)

    if (is_discrete):

        d_observation = env.observation_space.shape[0]
        d_action = env.action_space.n

        interface["value"] = (d_observation, 1)
        interface["qvalue"] = (d_observation, d_action)
        interface["policy"] = (d_observation, d_action)

    elif (is_continuous):

        d_observation = env.observation_space.shape[0]
        d_action = env.action_space.shape[0]

        interface["value"] = (d_observation, 1)
        interface["qvalue"] = (d_observation + d_action, 1)
        interface["policy"] = (d_observation + d_action, 1)

    return interface


