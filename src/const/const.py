import os
from enum import Enum
import gym

PATH_TO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH_TO_NETWORK_CONFIG = os.path.join(PATH_TO_ROOT, "network", "config")
PATH_TO_NETWORK_MODEL = os.path.join(PATH_TO_ROOT, "network", "model")

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
