#### Environment ####

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import torch
import gym

from ..const import SpaceType

class BaseEnvironment(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        self.state_space = None
        self.action_space = None
        self.observation_space = None
        self.state = None
        self._is_available = False
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self
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
        info_history
    ):
        raise NotImplementedError

class Environment(BaseEnvironment):

    def __init__(
        self
    ):
       super().__init__()
    
    def reset(
        self
    ):
        observation = self.state = self.observation_space.sample()
        return observation

    def setup(
        self,
        observation_type = SpaceType.CONTINUOUS,
        action_type = SpaceType.DISCRETE
    ):
        if ((type(observation_type) is SpaceType) and (type(action_type) is SpaceType)):
            
            if (observation_type == SpaceType.CONTINUOUS):
                self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,))
                self.state_space = self.observation_space
                self.state = None

            elif (observation_type == SpaceType.DISCRETE):
                self.observation_space = gym.spaces.Discrete(2)
                self.state_space = self.observation_space
                self.state = None

            if (action_type == SpaceType.CONTINUOUS):
                self.action_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,))

            elif (action_type == SpaceType.DISCRETE):
                self.action_space = gym.spaces.Discrete(2)
            
        self._become_available()
    
    def step(
        self,
        action
    ):
        warnings.warn("`step` is not implemented.")
        observation = self.observation_space.sample()
        reward = np.random.choice([0.0, 1.0])
        done = np.random.choice([True, False])
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
        info_history
    ):
        warnings.warn("`score` is not implemented.")
        score_dictionary = {}
        return score_dictionary

class GymEnvironment(BaseEnvironment):

    def __init__(
        self,
        name = None
    ):
        self.env = None
        self._is_available = False
        self.setup(name)

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
        observation = self.state = self.env.reset()
        return observation

    def setup(
        self,
        name = ""
    ):
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
        return (observation, reward, done, info)

    def sample(
        self
    ):
        action = self.env.sample()
        return action

class CartPoleEnvironment(GymEnvironment):

    def __init__(
        self
    ):
        super().__init__(
            name = "CartPole-v0"
        )
        self._t = 0
        self._T = 200

    def reset(
        self
    ):
        self._t = 0
        return super().reset()
    
    def step(
        self,
        action
    ):
        observation, reward, done, info = self.env.step(action)
        self._t += 1
        if (done):
            reward = 1.0 if (self._t >= self._T) else -1.0
        else:
            reward = 0.1
        return observation, reward, done, info
    
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
        return super().reset()
    
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
