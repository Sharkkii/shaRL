#### Environment (Mixin) ####

import random
import numpy as np
import torch

from .base import ModelBase
from ..environment import EnvironmentBase


class ModelMixin(ModelBase):

    def declare(self):
        self._env = None
        self._configuration = None
    
    @property
    def env(self): return self._env
    @property
    def interface(self): return self.env.interface
    @property
    def configuration(self): return self._configuration
    @property
    def state_space(self): return self.env.state_space
    @property
    def action_space(self): return self.env.action_space
    @property
    def observation_space(self): return self.env.observation_space

    def __init__(
        self,
        env,
        configuration = None,
        allow_setup = True
    ):
        ModelMixin.declare(self)
        if (allow_setup):
            ModelMixin.setup(
                self,
                env = env,
                configuration = configuration
            )

    def setup(
        self,
        env,
        configuration = None
    ):
        if ((env is None) or (configuration is None)):
            return
        if (not isinstance(env, EnvironmentBase)):
            raise ValueError("`env` must be 'Environment' object.")
        if (type(configuration) is not dict):
            raise ValueError("`configuration` must be 'Dictionary' object.")
        self._env = env
        self._configuration = configuration

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

    def update(
        self,
        history
    ):
        pass
