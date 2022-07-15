#### Environment (Mixin) ####

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EmptyModelBase
from .base import ModelBase
from .base import ApproximateModelBase
from ..env import EnvironmentBase


class EmptyModelMixin(EmptyModelBase):

    def __init__(self): raise NotImplementedError
    def setup(self): raise NotImplementedError
    def reset(self): raise NotImplementedError
    def step(self): raise NotImplementedError
    def sample(self): raise NotImplementedError
    def update(self): raise NotImplementedError

    @property
    def env(self): raise NotImplementedError
    @property
    def interface(self): raise NotImplementedError
    @property
    def configuration(self): raise NotImplementedError
    @property
    def state(self): raise NotImplementedError
    @property
    def state_space(self): raise NotImplementedError
    @property
    def action_space(self): raise NotImplementedError
    @property
    def observation_space(self): raise NotImplementedError


class ModelMixin(ModelBase):

    def declare(self):
        self._env = None
        self._configuration = None
        self._state = None
    
    @property
    def env(self): return self._env
    @property
    def interface(self): return self.env.interface
    @property
    def configuration(self): return self._configuration
    @property
    def state(self): return self._state
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
        self._state = self.env.reset()

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


class ApproximateModelMixin(ModelMixin, ApproximateModelBase):
    pass


class ApproximateForwardDynamicsModelMixin(ApproximateModelMixin):

    @property
    def forward_dynamics_model(self): return self._forward_dynamics_model
    @property
    def forward_dynamics_optimizer(self): return self._forward_dynamics_optimizer

    def declare(
        self
    ):
        self._forward_dynamics_model = None
        self._forward_dynamics_optimizer = None

    def __init__(
        self,
        env,
        configuration = None,
        allow_setup = True
    ):
        ApproximateModelMixin.__init__(
            self,
            env,
            configuration = configuration
        )
        ApproximateForwardDynamicsModelMixin.declare(self)
        if (allow_setup):
            ApproximateForwardDynamicsModelMixin.setup(
                self,
                env,
                configuration = configuration
            )
    
    def setup(
        self,
        env,
        configuration = None
    ):
        d_observation = self.env.observation_space.shape[0]
        d_action = self.env.action_space.n
        self._forward_dynamics_model = NNRegressor(din = d_observation + d_action, dout = d_observation)
        self._forward_dynamics_optimizer = torch.optim.Adam(self._forward_dynamics_model.parameters(), lr=1e-3)

    def step_wrapper(
        step
    ):
        def wrapper(
            self,
            action
        ):
            _, reward, done, info = step(self, action)
            d_action = self.env.action_space.n
            action = torch.eye(d_action)[action]
            next_state = self.eval_forward_dynamics(
                state = self.state,
                action = action
            )
            return next_state, reward, done, info
        return wrapper

    @step_wrapper
    def step(
        self,
        action
    ):
        return ApproximateModelMixin.step(self, action)

    def eval_forward_dynamics(
        self,
        state,
        action
    ):
        if (type(action) is int):
            d_action = self.env.action_space.n
            action = torch.eye(d_action)[action]
        next_state = self.forward_dynamics_model(torch.cat([state, action], dim=0))
        return next_state


class ApproximateInverseDynamicsModelMixin(ApproximateModelMixin):

    @property
    def inverse_dynamics_model(self): return self._inverse_dynamics_model
    @property
    def inverse_dynamics_optimizer(self): return self._inverse_dynamics_optimizer

    def declare(
        self
    ):
        self._inverse_dynamics_model = None
        self._inverse_dynamics_optimizer = None

    def __init__(
        self,
        env,
        configuration = None,
        allow_setup = True
    ):
        ApproximateModelMixin.__init__(
            self,
            env,
            configuration = configuration
        )
        ApproximateInverseDynamicsModelMixin.declare(self)
        if (allow_setup):
            ApproximateInverseDynamicsModelMixin.setup(
                self,
                env,
                configuration = configuration
            )
    
    def setup(
        self,
        env,
        configuration = None
    ):
        d_observation = self.env.observation_space.shape[0]
        d_action = self.env.action_space.n
        self._inverse_dynamics_model = NNClassifier(din = d_observation * 2, dout = d_action)
        self._inverse_dynamics_optimizer = torch.optim.Adam(self._inverse_dynamics_model.parameters(), lr=1e-3)

    def eval_inverse_dynamics(
        self,
        state,
        next_state
    ):
        action = self.inverse_dynamics_model(torch.cat([state, next_state], dim=0))
        return action


class NNClassifier(nn.Module):

    def __init__(
        self,
        din,
        dout
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_features = din,
            out_features = dout
        )

    def forward(
        self,
        x
    ):
        if (x.ndim == 1):
            x = x.unsqueeze(0)
        y = self.linear(x)
        y = F.softmax(y, dim=1)
        return y


class NNRegressor(nn.Module):

    def __init__(
        self,
        din,
        dout
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_features = din,
            out_features = dout
        )

    def forward(
        self,
        x
    ):
        if (x.ndim == 1):
            x = x.unsqueeze(0)
        y = self.linear(x)
        return y
