import pytest
import numpy as np
import gym
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.environment import Environment
from src.environment import GymEnvironment

@pytest.mark.L2
class TestEnvironment():

    @pytest.mark.unit
    def test_should_be_unavailable_on_initialization(self):
        env = Environment()
        assert env.is_available == False

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        env = Environment()
        env.setup()
        assert env.is_available == True

    @pytest.mark.unit
    def test_should_have_observation_space(self):
        env = Environment()
        env.setup()
        assert isinstance(env.observation_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_action_space(self):
        env = Environment()
        env.setup()
        assert isinstance(env.action_space, gym.Space)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_observation_space, expected_observation_type",
        [
            (gym.spaces.Box(0, 1, shape=(1,)), SpaceType.CONTINUOUS),
            (gym.spaces.Discrete(2), SpaceType.DISCRETE)
        ]
    )
    @pytest.mark.parametrize(
        "input_action_space, expected_action_type",
        [
            (gym.spaces.Box(0, 1, shape=(1,)), SpaceType.CONTINUOUS),
            (gym.spaces.Discrete(2), SpaceType.DISCRETE)
        ]
    )
    def test_spaces_can_be_configured(
        self,
        input_observation_space, input_action_space,
        expected_observation_type, expected_action_type
    ):
        env = Environment()
        env.setup(
            observation_space = input_observation_space,
            action_space = input_action_space
        )
        assert env.interface.observation_type is expected_observation_type
        assert env.interface.action_type is expected_action_type

    @pytest.mark.unit
    def test_reset_method_should_return_single_observation(self):
        env = Environment()
        env.setup()
        observation = env.reset()
        assert type(observation) is np.ndarray

    @pytest.mark.unit
    def test_sample_method_should_return_single_action(self):
        env = Environment()
        env.setup()
        _ = env.reset()
        action = env.sample()
        assert action is not None
    
    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_r_done_info(self):
        env = Environment()
        env.setup()
        action = env.sample()
        observation, reward, done, info = env.step(action)
        assert type(observation) is np.ndarray
        assert type(reward) in (np.float32, np.float64)
        assert type(done) in (bool, np.bool_)

@pytest.mark.L2
class TestGymEnvironment():

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization(self):
        env = GymEnvironment()
        assert env.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_valid_initialization(self):
        env = GymEnvironment(
            name = "CartPole-v1"
        )
        assert env.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_on_invalid_intialization(self):
        with pytest.raises(ValueError) as message:
            env = GymEnvironment(
                name = "INVALID_NAME"
            )

    @pytest.mark.unit
    def test_should_be_available_after_valid_setup(self):
        env = GymEnvironment()
        env.setup(
            name = "CartPole-v1"
        )
        assert env.is_available == True

    def test_should_raise_value_error_after_invalid_setup(self):
        env = GymEnvironment()
        with pytest.raises(ValueError) as message:
            env = GymEnvironment(
                name = "INVALID_NAME"
            )

    @pytest.mark.unit
    def test_should_have_observation_space(self):
        env = Environment()
        env.setup()
        assert isinstance(env.observation_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_action_space(self):
        env = Environment()
        env.setup()
        assert isinstance(env.action_space, gym.Space)

    @pytest.mark.unit
    def test_reset_method_should_return_single_observation(self):
        env = Environment()
        env.setup()
        observation = env.reset()
        assert type(observation) is np.ndarray

    @pytest.mark.unit
    def test_sample_method_should_return_single_action(self):
        env = Environment()
        env.setup()
        _ = env.reset()
        action = env.sample()
        assert action is not None

    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_r_done_info(self):
        env = Environment()
        env.setup()
        action = env.sample()
        observation, reward, done, info = env.step(action)
        assert type(observation) is np.ndarray
        assert type(reward) in (np.float32, np.float64)
        assert type(done) in (bool, np.bool_)
