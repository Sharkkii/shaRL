import os
from enum import Enum
import gym

PATH_TO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH_TO_NETWORK_CONFIG = os.path.join(PATH_TO_ROOT, "network", "config")
PATH_TO_NETWORK_MODEL = os.path.join(PATH_TO_ROOT, "network", "model")

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

class EnvironmentModelType(Enum):
    NONE = 0
    MODEL_BASED = 1
    MODEL_FREE = 2

class AgentStrategyType(Enum):
    NONE = 0
    VALUE_BASED = 1
    POLICY_BASED = 2
    ACTOR_CRITIC = 3

class AgentBehaviorType(Enum):
    NONE = 0
    ON_POLICY = 1
    OFF_POLICY = 2

class AgentLearningType(Enum):
    NONE = 0
    ONLINE = 1
    OFFLINE = 2

def cast_to_space_type(
    env
):
    if (type(env.action_space) is gym.spaces.Discrete):
        return SpaceType.DISCRETE
    elif (type(env.action_space) is gym.spaces.Box):
        return SpaceType.CONTINUOUS
    else:
        return SpaceType.NONE
