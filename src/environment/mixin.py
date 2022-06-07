#### Environment (Mixin) ####

import random
import numpy as np
import torch
import gym
from ..const import SpaceType
from ..common import Component

from .base import EnvironmentBase
from .base import GymEnvironmentBase
from .base import GoalReachingTaskEnvironmentBase
from .helper import get_environment_interface


class EnvironmentMixin(EnvironmentBase, Component):
    
    def declare(self):
        self._interface = None
        self._configuration = None
        self._state_space = None
        self._action_space = None
        self._observation_space = None
        self._state = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def state_space(self): return self._state_space
    @property
    def action_space(self): return self._action_space
    @property
    def observation_space(self): return self._observation_space
    @property
    def state(self): return self._state

    def __init__(
        self,
        configuration = None,
        allow_setup = True
    ):
        EnvironmentMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            EnvironmentMixin.setup(self, configuration = configuration)
        
    def setup(
        self,
        configuration = None
    ):
        if (configuration is None):
            return
        if (type(configuration) is not dict):
            raise ValueError("`configuration` must be 'Dictionary' object.")
        if ("observation_space" not in configuration):
            raise ValueError("`configuration` must have 'observation_space' key.")
        if ("action_space" not in configuration):
            raise ValueError("`configuration` must have 'action_space' key.")
        observation_space = configuration["observation_space"]
        action_space = configuration["action_space"]

        if (isinstance(observation_space, gym.spaces.Space) and isinstance(action_space, gym.spaces.Space)):      
            self._interface = get_environment_interface(
                observation_space = observation_space,
                action_space = action_space
            )
            self._configuration = configuration
            self._state_space = observation_space
            self._action_space = action_space
            self._observation_space = observation_space
            self._state = None
            self._become_available()
    
    def reset(
        self
    ):
        observation = self.observation_space.sample()
        observation = torch.from_numpy(observation.astype(np.float32))
        self._state = observation
        return observation

    def step(
        self,
        action
    ):
        observation = self.observation_space.sample()
        observation = torch.from_numpy(observation.astype(np.float32))
        reward = np.random.rand()
        reward = torch.tensor(reward, dtype=torch.float32)
        done = random.choice([True, False])
        info = None
        return observation, reward, done, info

    def sample(
        self
    ):
        action = self.action_space.sample()
        return action

    def score(
        self,
        history,
        info_history = None
    ):
        score_dictionary = {}
        return score_dictionary

    def can_accept_action(
        self,
        action
    ):
        flag = False
        if (not self.is_available):
            return flag

        if (type(action) is torch.Tensor):
            if (self.interface.action_type is SpaceType.DISCRETE):
                flag = (action.dtype is torch.long) and (tuple(action.shape) in ((), (1,)))
            elif (self.interface.action_type is SpaceType.CONTINUOUS):
                flag = (action.dtype is torch.float32) and (tuple(action.shape) == self.interface.action_shape)
        
        return flag


class GymEnvironmentMixin(EnvironmentMixin, GymEnvironmentBase):

    def declare(self):
        self._env = None

    @property
    def env(self): return self._env
    
    def __init__(
        self,
        configuration = None
    ):
        EnvironmentMixin.__init__(
            self,
            configuration = configuration,
            allow_setup = False
        )
        GymEnvironmentMixin.declare(self)
        GymEnvironmentMixin.setup(
            self,
            configuration = configuration
        )

    def setup(
        self,
        configuration = None
    ):
        if (configuration is None):
            return
        if (type(configuration) is not dict):
            raise ValueError("`configuration` must be 'Dictionary' object.")
        if ("name" not in configuration):
            raise ValueError("`configuration` must have 'name' key.")
        name = configuration["name"]
        if (type(name) is str):
            try:
                self._env = gym.make(name)
                EnvironmentMixin.setup(
                    self,
                    configuration = {
                        "observation_space": self.env.observation_space,
                        "action_space": self.env.action_space
                    }
                )
            except (gym.error.Error, gym.error.NameNotFound) as e:
                raise ValueError(e)

    def reset_decorator(reset):
        def wrapper(
            self
        ):
            observation = reset(self)
            observation = torch.from_numpy(observation.astype(np.float32))
            # self._state = observation
            return observation
        return wrapper

    @reset_decorator
    def reset(self):
        observation = self.env.reset()
        return observation

    def step_decorator(step):
        def wrapper(
            self,
            action
        ):
            observation, reward, done, info = step(self, action)
            observation = torch.from_numpy(observation.astype(np.float32))
            reward = torch.tensor(reward, dtype=torch.float32)
            return (observation, reward, done, info)
        return wrapper

    @step_decorator
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return (observation, reward, done, info)
        

class GoalReachingTaskEnvironmentMixin(EnvironmentMixin, GoalReachingTaskEnvironmentBase):
    
    def declare(self):
        self._goal_space = None
        self._goal_state = None
    
    @property
    def goal_space(self): return self._goal_space
    @property
    def goal_state(self): return self._goal_state

    def __init__(
        self,
        configuration = None
    ):
        super().__init__(configuration = configuration)
        GoalReachingTaskEnvironmentMixin.declare(self)
        GoalReachingTaskEnvironmentMixin.setup(self, configuration = configuration)

    def setup(
        self,
        configuration = None
    ):
        if (configuration is None):
            return
        if (type(configuration) is not dict):
            raise ValueError("`configuration` must be 'Dictionary' object.")
        if ("goal_space" not in configuration):
            raise ValueError("`configuration` must have 'goal_space' key.")
        goal_space = configuration["goal_space"]

        if (isinstance(goal_space, gym.spaces.Space)):
            self._goal_space = goal_space
            # self._goal_state = None
            self._become_available()

    def reset_decorator(reset):
        def wrapper(
            self,
            use_goal = True
        ):
            goal = self.goal_space.sample()
            self._goal_state = torch.from_numpy(goal.astype(np.float32))
            observation = reset(self)
            if (use_goal):
                return (observation, self.goal_state)
            else:
                return observation
        return wrapper

    def step_decorator(step):
        def wrapper(
            self,
            action,
            use_goal = True,
            use_reward = False
        ):
            observation, reward, done, info = step(self, action)
            if (use_goal and use_reward):
                return (observation, self.goal_state, reward, done, info)
            elif (use_goal and (not use_reward)):
                return (observation, self.goal_state, done, info)
            elif ((not use_goal) and use_reward):
                return (observation, reward, done, info)
            else:
                return (observation, done, info)
        return wrapper
