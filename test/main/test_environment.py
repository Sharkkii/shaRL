import pytest
import numpy as np
import torch
import gym
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.environment import Environment
from src.environment import GoalReachingTaskEnvironment
from src.environment import GymEnvironment


default_gymenvironment_configuration = { "name": "CartPole-v1" }

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
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32

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
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32
        assert type(reward) is torch.Tensor
        assert reward.dtype is torch.float32
        assert type(done) is bool

@pytest.mark.L2
class TestGymEnvironment():

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization(self):
        env = GymEnvironment()
        assert env.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_valid_initialization(self):
        configuration = { "name": "CartPole-v1" }
        env = GymEnvironment(
            configuration = configuration
        )
        assert env.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "configuration",
        [
            { "name": "INVALID_NAME" },
            { "key": "value" }
        ]
    )
    def test_should_raise_value_error_on_invalid_intialization(self, configuration):
        with pytest.raises(ValueError) as message:
            env = GymEnvironment(
                configuration = configuration
            )

    @pytest.mark.unit
    def test_should_be_available_after_valid_setup(self):
        configuration = { "name": "CartPole-v1" }
        env = GymEnvironment()
        env.setup(
            configuration = configuration
        )
        assert env.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "configuration",
        [
            { "name": "INVALID_NAME" },
            { "key": "value" }
        ]
    )
    def test_should_raise_value_error_after_invalid_setup(self, configuration):
        configuration = configuration
        env = GymEnvironment()
        with pytest.raises(ValueError) as message:
            env = GymEnvironment(
               configuration = configuration
            )

    @pytest.mark.unit
    def test_should_have_observation_space(self):
        env = GymEnvironment()
        env.setup(
            configuration = default_gymenvironment_configuration
        )
        assert isinstance(env.observation_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_action_space(self):
        env = GymEnvironment()
        env.setup(
            configuration = default_gymenvironment_configuration
        )
        assert isinstance(env.action_space, gym.Space)

    @pytest.mark.unit
    def test_reset_method_should_return_single_observation(self):
        env = GymEnvironment()
        env.setup(
            configuration = default_gymenvironment_configuration
        )
        observation = env.reset()
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32

    @pytest.mark.unit
    def test_sample_method_should_return_single_action(self):
        env = GymEnvironment()
        env.setup(
            configuration = default_gymenvironment_configuration
        )
        _ = env.reset()
        action = env.sample()
        assert action is not None

    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_r_done_info(self):
        env = GymEnvironment()
        env.setup(
            configuration = default_gymenvironment_configuration
        )
        _ = env.reset()
        action = env.sample()
        observation, reward, done, info = env.step(action)
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32
        assert type(reward) is torch.Tensor
        assert reward.dtype is torch.float32
        assert type(done) is bool


@pytest.mark.L2
class TestGoalReachingTaskEnvironment():

    @pytest.mark.unit
    def test_should_be_unavailable_on_initialization(self):
        env = GoalReachingTaskEnvironment()
        assert env.is_available == False

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        assert env.is_available == True

    @pytest.mark.unit
    def test_should_have_observation_space(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        assert isinstance(env.observation_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_action_space(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        assert isinstance(env.action_space, gym.Space)

    @pytest.mark.unit
    def test_should_have_goal_space(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        assert isinstance(env.goal_space, gym.Space)

    @pytest.mark.unit
    def test_reset_method_should_return_single_observation_if_do_not_use_goal(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        observation = env.reset(
            use_goal = False
        )
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32

    @pytest.mark.unit
    def test_reset_method_should_return_observation_and_goal_if_only_use_goal(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        observation, goal = env.reset(
            use_goal = True
        )
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32
        assert type(goal) is torch.Tensor
        assert goal.dtype is torch.float32

    @pytest.mark.unit
    def test_sample_method_should_return_single_action(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        _ = env.reset()
        action = env.sample()
        assert action is not None
    
    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_done_info_if_dont_use_goal(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        _ = env.reset()
        action = env.sample()
        observation, done, info = env.step(
            action,
            use_goal = False
        )
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32
        assert type(done) is bool

    @pytest.mark.unit
    def test_step_method_should_return_tuple_of_obs_goal_done_info_if_only_use_goal(self):
        env = GoalReachingTaskEnvironment()
        env.setup()
        _ = env.reset()
        action = env.sample()
        observation, goal, done, info = env.step(
            action,
            use_goal = True
        )
        assert type(observation) is torch.Tensor
        assert observation.dtype is torch.float32
        assert type(goal) is torch.Tensor
        assert goal.dtype is torch.float32
        assert type(done) is bool
