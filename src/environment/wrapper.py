#### Environment ####

import math
import numpy as np
import torch
import gym
from gym import spaces
# from gym.wrappers import TimeLimit


# transparent wrapper
class TransparentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        return self.env.step(action)


# wrapper for CartPole-v0 environment
class CartPoleWrapper(gym.Wrapper):
    
    def __init__(self, env, positive_scale=1.0, negative_scale=1.0):
        super().__init__(env)
        assert(positive_scale > 0)
        assert(negative_scale > 0)
        self.positive_scale = positive_scale
        self.negative_scale = negative_scale
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = self.positive_scale if (not done) else - self.negative_scale
        return observation, reward, done, info


