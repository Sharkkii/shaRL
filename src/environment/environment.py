#### Environment ####

import math
import numpy as np
import torch
import gym
from gym import spaces
# from gym.wrappers import TimeLimit


# pseudo MountainCar-v0 environment
# see https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
class PseudoMountainCar(gym.Env):

    def __init__(
        self,
        goal_velocity=0,
        x_max_speed=5,
        x_force=5,
        x_gravity=5,
        goal_reward=10.0
    ):

        self.t = 0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07 * x_max_speed
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001 * x_force
        self.gravity = 0.0025 * x_gravity

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )
        self.reward_range = [-1.0, goal_reward]
        self.goal_reward = goal_reward

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.t += 1

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = -1.0
        # reward = 10.0 if done else position - self.max_position
        # reward = self.goal_reward if done else self._height(position) - 1.0

        if (self.t >= 200):
            self.t = 0
            done = True

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.t = 0
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55
