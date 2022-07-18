#### Environment ####

import warnings
import random
import numpy as np
import torch
import gym

from .base import EnvironmentBase
from .base import GoalReachingTaskEnvironmentBase
from .mixin import EnvironmentMixin
from .mixin import GymEnvironmentMixin
from .mixin import GoalReachingTaskEnvironmentMixin


class Environment(EnvironmentMixin, EnvironmentBase):

    def __init__(
        self,
        configuration = None
    ):
        super().__init__(configuration = configuration)


class GymEnvironment(GymEnvironmentMixin, EnvironmentBase):

    def __init__(self, configuration = None):
        super().__init__(configuration = configuration)


class GoalReachingTaskEnvironment(GoalReachingTaskEnvironmentMixin, EnvironmentBase):

    def __init__(self, configuration = None):
        super().__init__(configuration = configuration)

    @GoalReachingTaskEnvironmentMixin.reset_decorator
    def reset(self):
        observation = self.observation_space.sample()
        observation = torch.from_numpy(observation.astype(np.float32))
        return observation

    @GoalReachingTaskEnvironmentMixin.step_decorator
    def step(self, action):
        observation = self.observation_space.sample()
        observation = torch.from_numpy(observation.astype(np.float32))
        reward = np.random.rand()
        reward = torch.tensor(reward, dtype=torch.float32)
        done = random.choice([True, False])
        info = None
        return observation, reward, done, info


class CartPoleEnvironment(GymEnvironmentMixin, EnvironmentBase):

    def declare(self):
        self.name = None
        self._t = None
        self._T = None

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
        CartPoleEnvironment.declare(self)
        CartPoleEnvironment.setup(self, version)

    def setup(self, version):
        if (version not in ("v0", "v1")):
            raise ValueError("`version`: ('v0', 'v1')")
        self._t = 0
        self._T = 500 if (version == "v1") else 200
    
    def reset(self):
        self._t = 0
        observation = super().reset()
        return observation
    
    def step(
        self,
        action
    ):
        self._t += 1
        observation, reward, done, info = super().step(action)

        if (done):
            reward = torch.tensor(1.0, dtype=torch.float32) if (self._t >= self._T) else torch.tensor(-1.0, dtype=torch.float32)
        else:
            reward = torch.tensor(1.0, dtype=torch.float32)
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


class DiscreteMountainCarEnvironment(GymEnvironmentMixin, GoalReachingTaskEnvironmentMixin, EnvironmentBase):

    def declare(self):
        self.name = None
        self.reward_spec = None
        self._t = None
        self._T = None

    def __init__(
        self,
        reward_spec = "sparse",
        difficulty = "easy"
    ):
        if (reward_spec not in ("sparse", "dense")):
            raise ValueError("`reward_spec`: ('sparse', 'dense')")
        if (difficulty not in ("easy", "normal", "hard")):
            raise ValueError("`difficulty`: ('easy', 'normal', 'hard'")
        self.name = "MountainCar-v0"

        GymEnvironmentMixin.__init__(
            self,
            configuration = { "name": self.name }
        )

        observation_space = self.env.observation_space
        goal_space = gym.spaces.Box(
            np.array([self.env.goal_position, self.env.goal_velocity]),
            np.array([self.env.max_position, self.env.max_speed]),
            dtype = np.float32
        )
        action_space = self.env.action_space
        GoalReachingTaskEnvironmentMixin.__init__(
            self,
            configuration = {
                "observation_space": observation_space,
                "action_space": action_space,
                "goal_space": goal_space
            }
        )
        DiscreteMountainCarEnvironment.declare(self)
        DiscreteMountainCarEnvironment.setup(
            self,
            reward_spec = reward_spec,
            difficulty = difficulty
        )

    def setup(
        self,
        reward_spec = "sparse",
        difficulty = "easy"
    ):
        if (reward_spec not in ("sparse", "dense")):
            raise ValueError("`reward_spec`: ('sparse', 'dense')")
        if (reward_spec == "dense"):
            raise NotImplementedError("`reward_spec = dense` is not supported right now.")

        if (difficulty not in ("easy", "normal", "hard")):
            raise ValueError("`difficulty`: ('easy', 'normal', 'hard'")
        if (difficulty == "easy"):
            C = 5.0
        elif (difficulty == "normal"):
            C = 2.5
        elif (difficulty == "hard"):
            C = 1.0

        self.reward_spec = reward_spec
        self.difficulty = difficulty
        self._t = 0
        self._T = 200
        self.env.env.env.force = self.env.force * C
        self.env.env.env.gravity = self.env.gravity / C
        self.env.env.env.max_speed = self.env.max_speed * C
    
    @GoalReachingTaskEnvironmentMixin.reset_decorator
    def reset(
        self
    ):
        self._t = 0
        observation = GymEnvironmentMixin.reset(self)
        return observation

    @GoalReachingTaskEnvironmentMixin.step_decorator
    def step(
        self,
        action
    ):
        self._t += 1
        observation, reward, done, info = GymEnvironmentMixin.step(self, action)
        return (observation, reward, done, info)

    def score(
        self,
        history,
        info_history = None
    ):
        score_dictionary = {
            "total_loss": None,
            "elapsed_time": None,
        }
        if (len(history) > 0):
            total_loss = 0
            for sgasg in history:
                pos_s = sgasg.state[0]
                pos_g = sgasg.goal[0]
                total_loss = total_loss + (pos_s - pos_g) ** 2
            elapsed_time = len(history)
            score_dictionary["total_loss"] = total_loss
            score_dictionary["elapsed_time"] = elapsed_time
        
        return score_dictionary


class ContinuousMountainCarEnvironment(GymEnvironmentMixin, GoalReachingTaskEnvironmentMixin, GoalReachingTaskEnvironmentBase):

    def declare(self):
        self.name = None
        self._t = None
        self._T = None

    def __init__(
        self
    ):
        super().__init__(
            configuration = { "name": "MountainCarContinuous-v0" }
        )
        ContinuousMountainCarEnvironment.declare(self)
        ContinuousMountainCarEnvironment.setup(self)

    def setup(self):
        self._t = 0
        self._T = 200

    @GoalReachingTaskEnvironmentMixin.reset_decorator
    def reset(self):
        self._t = 0
        observation = super().reset()
        return observation

    @GoalReachingTaskEnvironmentMixin.step_decorator
    def step(
        self,
        action
    ):
        self._t += 1
        observation, reward, done, info = super().step(action)
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


class PendulumEnvironment(GymEnvironmentMixin, EnvironmentBase):

    def declare(self):
        self.name = None
        self._t = None
        self._T = None

    def __init__(
        self
    ):
        super().__init__(
            configuration = { "name": "Pendulum-v1" }
        )
        PendulumEnvironment.declare(self)
        PendulumEnvironment.setup(self)

    def setup(self):
        self._t = 0
        self._T = 200

    def reset(self):
        self._t = 0
        observation = super().reset()
        return observation
    
    def step(
        self,
        action
    ):
        self._t += 1
        observation, reward, done, info = super().step(action)
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


# for compatilibity (will be removed)
class BaseEnvironment():
    pass