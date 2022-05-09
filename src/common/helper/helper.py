import gym
from functools import reduce
from ...const import SpaceType

def cast_space_to_type(
    space
):
    if (type(space) is gym.spaces.Discrete):
        return SpaceType.DISCRETE
    elif (type(space) is gym.spaces.Box):
        return SpaceType.CONTINUOUS
    else:
        return SpaceType.NONE

def _is_tuple_of_int(argument):
    if (type(argument) is tuple):
        return reduce(lambda acc, x: acc and (x is int), argument, True)
    else:
        return False