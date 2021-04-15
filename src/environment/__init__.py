from .environment import PseudoMountainCar
from .model import Model
from .model import DefaultModel
from .wrapper import TransparentWrapper
from .wrapper import CartPoleWrapper

import gym
from gym.envs.registration import register
register(
    id="PseudoMountainCar-v0",
    entry_point="environment:PseudoMountainCar"
)
