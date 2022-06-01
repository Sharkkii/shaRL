#### Environment ####

from abc import ABCMeta, abstractmethod
import warnings
import random
import numpy as np
import torch
import gym

from .helper import get_environment_interface
from ..const import SpaceType

class BaseEnvironment(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        configuration = None
    ):
        self.state_space = None
        self.action_space = None
        self.observation_space = None
        self.state = None
        self._is_available = False
        self.interface = None
        self.configuration = None
    
    @abstractmethod
    def reset(
        self
    ):
        return None

    @abstractmethod
    def setup(
        self,
        configuration = None
    ):
        self._become_available()

    @property
    def is_available(
        self
    ):
        return self._is_available

    def _become_available(
        self
    ):
        self._is_available = True

    def _become_unavailable(
        self
    ):
        self._is_available = False

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
    
    @abstractmethod
    def step(
        self,
        action
    ):
        raise NotImplementedError
    
    # @abstractmethod
    def sample(
        self
    ):
        raise NotImplementedError

    # @abstractmethod
    def update(
        self
    ):
        raise NotImplementedError

    # @abstractmethod
    def score(
        self,
        history,
        info_history = None
    ):
        raise NotImplementedError


class Environment(BaseEnvironment):

    def __init__(
        self,
        configuration = None
    ):
        super().__init__(
            configuration = configuration
        )
    
    def reset(
        self
    ):
        observation = self.observation_space.sample()
        observation = torch.from_numpy(observation.astype(np.float32))
        self.state = observation
        return observation

    def setup(
        self,
        configuration = None,
        observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,)), # will be in `configuration`
        action_space = gym.spaces.Discrete(2) # will be in `configuration`
    ):
        if (isinstance(observation_space, gym.spaces.Space) and isinstance(action_space, gym.spaces.Space)):      
            self.observation_space = self.state_space = observation_space
            self.action_space = action_space 
            self.state = None
            self.interface = get_environment_interface(
                self.observation_space,
                self.action_space
            )
            self._become_available()
    
    def step(
        self,
        action
    ):
        warnings.warn("`step` is not implemented.")
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
    
    def update(
        self
    ):
        warnings.warn("`update` cannot be used for an Environment instance.")

    def score(
        self,
        history,
        info_history = None
    ):
        warnings.warn("`score` is not implemented.")
        score_dictionary = {}
        return score_dictionary


class GoalReachingTaskEnvironment(Environment):

    def __init__(
        self,
        configuration = None
    ):
        super().__init__(
            configuration = configuration
        )
        self.goal_space = None
        self.goal = None

    def reset(
        self,
        use_goal = False
    ):
        goal = self.goal_space.sample()
        self.goal = torch.from_numpy(goal.astype(np.float32))
        observation = super().reset()
        if (use_goal):
            return (observation, self.goal)
        else:
            return observation
        
    def setup(
        self,
        configuration = None,
        observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,)), # will be in `configuration`
        action_space = gym.spaces.Discrete(2), # will be in `configuration`
        goal_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,)) # will be in `configuration`
    ):
        if (isinstance(observation_space, gym.spaces.Space) and isinstance(action_space, gym.spaces.Space) and isinstance(goal_space, gym.spaces.Space)):
            self.goal_space = goal_space
            self.observation_space = self.state_space = observation_space
            self.action_space = action_space
            self.goal = None
            self.state = None
            self.interface = get_environment_interface(
                self.observation_space,
                self.action_space
            )
            self._become_available()

    def step(
        self,
        action,
        use_goal = True,
        use_reward = False
    ):
        observation, reward, done, info = super().step(action)
        if (use_goal and use_reward):
            return (observation, self.goal, reward, done, info)
        elif (use_goal and (not use_reward)):
            return (observation, self.goal, done, info)
        elif ((not use_goal) and use_reward):
            return (observation, reward, done, info)
        else:
            return (observation, done, info)


class GymEnvironment(BaseEnvironment):

    def __init__(
        self,
        configuration = None
    ):
        self.env = None
        self._is_available = False
        self.setup(
            configuration = configuration
        )

    @property
    def observation_space(
        self
    ):
        return self.env.observation_space

    @property
    def action_space(
        self
    ):
        return self.env.action_space
        
    def reset(
        self
    ):
        observation = self.env.reset()
        observation = torch.from_numpy(observation.astype(np.float32))
        self.state = observation
        return observation

    def setup(
        self,
        configuration = None
    ):
        if (configuration is None): return
        if (type(configuration) is not dict):
            raise ValueError("`configuration` must be 'Dictionary' object.")
        if ("name" not in configuration):
            raise ValueError("`configuration` must have 'name' key.")
        
        name = configuration["name"]
        if (type(name) is str):
            try:
                self.env = gym.make(name)
                self._become_available()
            except gym.error.Error as e:
                raise ValueError(e)
            except gym.error.NameNotFound as e:
                raise ValueError(e)
    
    def step(
        self,
        action
    ):
        observation, reward, done, info = self.env.step(action)
        observation = torch.from_numpy(observation.astype(np.float32))
        reward = torch.tensor(reward, dtype=torch.float32)
        return (observation, reward, done, info)

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
        score_dictionary = {
            "total_reward": None
        }
        if (len(history) > 0):
            total_reward = 0
            for sars in history:
                total_reward = total_reward + sars.reward
            total_reward = int(total_reward)
            score_dictionary["total_reward"] = total_reward
        return score_dictionary

class CartPoleEnvironment(GymEnvironment):

    def __init__(
        self,
        version = "v1"
    ):
        if (version not in ("v0", "v1")):
            raise ValueError("`version`: ('v0', 'v1')")
        self.name = "CartPole-" + version

        super().__init__(
            configuration = { "name": self.name }
        )
        self.t = 0
        self.T = 500 if (version == "v1") else 200

    def reset(
        self
    ):
        self.t = 0
        return super().reset()
    
    def step(
        self,
        action
    ):
        self.t += 1
        observation, reward, done, info = self.env.step(action)

        if (done):
            reward = 1.0 if (self.t >= self.T) else -1.0
        else:
            reward = 1.0

        observation = torch.from_numpy(observation.astype(np.float32))
        reward = torch.tensor(reward, dtype=torch.float32)
        return (observation, reward, done, info)
    
    def score(
        self,
        history,
        info_history = None
    ):
        score_dictionary = {
            "duration": None
        }
        if (len(history) > 0):
            duration = len(history)
            score_dictionary["duration"] = duration
        return score_dictionary


class DiscreteMountainCarEnvironment(GymEnvironment):

    def __init__(
        self,
        reward_spec = "sparse"
    ):
        if (reward_spec not in ("sparse", "dense")):
            raise ValueError("`reward_spec`: ('sparse', 'dense')")
        self.name = "MountainCar-v0"
        self.reward_spec = reward_spec

        super().__init__(
            configuration = { "name": self.name }
        )
        self._t = 0
        self._T = 200
    
    def reset(
        self
    ):
        self._t = 0
        state = super().reset()
        return state

    def step(
        self,
        action
    ):
        self._t += 1
        observation, reward, done, info = self.env.step(action)
        if (self.reward_spec == "dense"):
            raise NotImplementedError("`reward_spec = dense` is not supported right now.")

        observation = torch.from_numpy(observation.astype(np.float32))
        reward = torch.tensor(reward, dtype=torch.float32)
        return (observation, reward, done, info)

    def score(
        self,
        history,
        info_history = None
    ):
        score_dictionary = {
            "total_reward": None
        }
        if (len(history) > 0):
            total_reward = 0
            for sars in history:
                total_reward = total_reward + sars.reward
            total_reward = int(total_reward)
            score_dictionary["total_reward"] = total_reward
        return score_dictionary


class ContinuousMountainCarEnvironment(GymEnvironment):

    def __init__(
        self
    ):
        super().__init__(
            name = "MountainCarContinuous-v0"
        )
        self._t = 0
        self._T = 200

    def reset(
        self
    ):
        self._t = 0
        state = super().reset()
        return state
    
    def step(
        self,
        action
    ):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
    
    def score(
        self,
        history,
        info_history = None
    ):
        score_dictionary = {
            "total_reward": None
        }
        if (len(history) > 0):
            total_reward = 0
            for (_, _, r, _) in history:
                total_reward = total_reward + r
            total_reward = int(total_reward)
            score_dictionary["total_reward"] = total_reward
        return score_dictionary

class PendulumEnvironment(GymEnvironment):

    def __init__(
        self
    ):
        super().__init__(
            name = "Pendulum-v1"
        )

    def reset(
        self
    ):
        return super().reset()
    
    def step(
        self,
        action
    ):
        observation, reward, done, info = self.env.step(action)
        reward *= 0.01
        return observation, reward, done, info

    def score(
        self,
        history,
        info_history = None
    ):
        score_dictionary = {
            "total_reward": None
        }
        if (len(history) > 0):
            total_reward = 0
            for (_, _, r, _) in history:
                total_reward = total_reward + r
            total_reward = int(total_reward)
            score_dictionary["total_reward"] = total_reward
        return score_dictionary
