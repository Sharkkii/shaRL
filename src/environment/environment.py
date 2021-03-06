#### Environment ####

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import torch
import gym


class BaseEnvironment(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def step(
        self,
        action
    ):
        raise NotImplementedError

    @abstractmethod
    def update(
        self
    ):
        raise NotImplementedError

    @abstractmethod
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
        self.state_space = None
        self.action_space = None
        self.observation_space = None
        self.state = None
    
    def reset(
        self
    ):
        observation = self.state = None
        return observation
    
    def step(
        self,
        action
    ):
        warnings.warn("`step` is not implemented.")
        observation = None
        reward = None
        done = True
        info = None
        return observation, reward, done, info
    
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

class GymEnvironment(Environment):

    def __init__(
        self,
        name = ""
    ):
        self.env = gym.make(name)
        self.state_space = self.env.observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state = None
        
    def reset(
        self
    ):
        observation = self.state = self.env.reset()
        return observation
    
    def step(
        self,
        action
    ):
        return self.env.step(action)

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
