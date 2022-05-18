import gym
from functools import reduce
from ...const import SpaceType
from ...dataset import Data

def cast_space_to_type(
    space
):
    if (type(space) is gym.spaces.Discrete):
        return SpaceType.DISCRETE
    elif (type(space) is gym.spaces.Box):
        return SpaceType.CONTINUOUS
    else:
        return SpaceType.NONE

def is_tuple_of_int(argument):
    if (type(argument) is tuple):
        return reduce(lambda acc, x: acc and (type(x) is int), argument, True)
    else:
        return False

def is_list_of_data(argument):
    if (type(argument) is list):
        return reduce(lambda acc, x: acc and (isinstance(x, Data)), argument, True)
    else:
        return False
