import pytest
import torch
import gym
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.environment import Model
from src.environment import EnvironmentBase
from src.environment import Environment


default_model_configuration = {}
default_environment_configuration = {
    "observation_space": gym.spaces.Box(low = 0.0, high = 1.0, shape=(1,)),
    "action_space": gym.spaces.Discrete(2)
}

@pytest.mark.L2
class TestModel():

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
    def test_should_have_env(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = Model(
            env = env,
            configuration = default_model_configuration
        )
        assert isinstance(model.env, EnvironmentBase)

    @pytest.mark.unit
    def test_should_have_observation_space(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = Model(
            env = env,
            configuration = default_model_configuration
        )
        assert isinstance(model.observation_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_action_space(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = Model(
            env = env,
            configuration = default_model_configuration
        )
        assert isinstance(model.action_space, gym.Space)

    @pytest.mark.unit
    def test_reset_method_should_return_single_observation(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = Model(
            env = env,
            configuration = default_model_configuration
        )
        observation = env.reset()
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32

    @pytest.mark.unit
    def test_sample_method_should_return_single_action(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = Model(
            env = env,
            configuration = default_model_configuration
        )
        _ = model.reset()
        action = model.sample()
        assert action is not None
    
    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_r_done_info(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        model = Model(
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
