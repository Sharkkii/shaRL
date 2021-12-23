from enum import Enum
import gym

class PhaseType(Enum):
    NONE = 0
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

class SpaceType(Enum):
    NONE = 0
    DISCRETE = 1
    CONTINUOUS = 2

class MeasureType(Enum):
    NONE = 0
    VALUE = 1
    QVALUE = 2
    POLICY = 3

def cast_to_space_type(
    env
):
    if (type(env.action_space) is gym.spaces.Discrete):
        return SpaceType.DISCRETE
    elif (type(env.action_space) is gym.spaces.Box):
        return SpaceType.CONTINUOUS
    else:
        return SpaceType.NONE