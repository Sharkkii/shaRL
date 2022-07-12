import pytest
import numpy as np
import torch
import gym
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.environment import EnvironmentBase
from src.environment import Environment
from src.environment import Model
from src.environment import ApproximateForwardDynamicsModel
from src.environment import ApproximateInverseDynamicsModel


default_model_configuration = {}
default_environment_configuration = {
    "observation_space": gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,)),
    "action_space": gym.spaces.Discrete(2)
}

@pytest.mark.L2
@pytest.mark.parametrize(
    "TModel",
    [ Model, ApproximateForwardDynamicsModel, ApproximateInverseDynamicsModel ]
)
class TestModel:

    # @pytest.mark.unit
    # def test_should_be_unavailable_on_empty_initialization(self):
    #     model = Model()
    #     assert model.is_available == False
    
    # @pytest.mark.unit
    # def test_should_be_available_on_nonempty_initialization(self):
    #     model = Model(configuration = default_model_configuration)
    #     assert model.is_available == True

    # @pytest.mark.unit
    # def test_should_be_available_after_setup(self):
    #     model = Model()
    #     model.setup(configuration = default_model_configuration)
    #     assert model.is_available == True

    @pytest.mark.unit
    def test_should_have_env(self, TModel):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = TModel(
            env = env,
            configuration = default_model_configuration
        )
        assert isinstance(model.env, EnvironmentBase)

    @pytest.mark.unit
    def test_should_have_observation_space(self, TModel):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = TModel(
            env = env,
            configuration = default_model_configuration
        )
        assert isinstance(model.observation_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_action_space(self, TModel):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = TModel(
            env = env,
            configuration = default_model_configuration
        )
        assert isinstance(model.action_space, gym.Space)

    @pytest.mark.unit
    def test_reset_method_should_return_single_observation(self, TModel):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = TModel(
            env = env,
            configuration = default_model_configuration
        )
        observation = model.reset()
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32

    @pytest.mark.unit
    def test_sample_method_should_return_single_action(self, TModel):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = TModel(
            env = env,
            configuration = default_model_configuration
        )
        _ = model.reset()
        action = model.sample()
        assert action is not None
    
    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_r_done_info(self, TModel):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = TModel(
            env = env,
            configuration = default_model_configuration
        )
        action = model.sample()
        observation, reward, done, info = model.step(action)
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32
        assert type(reward) is torch.Tensor
        assert reward.dtype is torch.float32
        assert type(done) is bool


class TestApproximateModel:

    @pytest.mark.unit
    def test_eval_forward_dynamics_method_should_return_single_state(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = ApproximateForwardDynamicsModel(
            env = env,
            configuration = default_model_configuration
        )
        state = env.observation_space.sample()
        state = torch.from_numpy(state.astype(np.float32))
        action = env.action_space.sample()
        next_state = model.eval_forward_dynamics(state, action)

        assert type(next_state) is torch.Tensor
        assert next_state.dtype is torch.float32

    @pytest.mark.unit
    def test_eval_inverse_dynamics_method_should_return_single_state(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = ApproximateInverseDynamicsModel(
            env = env,
            configuration = default_model_configuration
        )
        state = env.observation_space.sample()
        state = torch.from_numpy(state.astype(np.float32))
        next_state = env.observation_space.sample()
        next_state = torch.from_numpy(next_state.astype(np.float32))
        action = model.eval_inverse_dynamics(state, next_state)

        assert type(action) is torch.Tensor
        assert next_state.dtype is torch.float32
