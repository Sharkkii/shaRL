import pytest
import gym
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.environment import Environment
from src.environment import GoalReachingTaskEnvironment
from src.agent import Agent
from src.agent import GoalConditionedAgent
from src.controller import RLController
from src.controller import GoalConditionedRLController

default_environment_configuration = {
    "observation_space": gym.spaces.Box(0, 1, shape=(1,)),
    "action_space": gym.spaces.Discrete(2),
    "goal_space": gym.spaces.Box(0, 1, shape=(1,))
}

default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

default_goal_conditioned_agent_interface = AgentInterface(
    sin = 1 * 2,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

@pytest.mark.L1
class TestController():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        controller = RLController()
        assert controller.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        env = Environment()
        agent = Agent()
        controller = RLController(
            environment = env,
            agent = agent
        )
        assert controller.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        env = Environment()
        agent = Agent()
        controller = RLController()
        controller.setup(
            environment = env,
            agent = agent
        )
        assert controller.is_available == True

    @pytest.mark.unit
    @pytest.mark.integration
    def test_train_method_should_work(self):
        env = Environment(
            configuration = default_environment_configuration
        )
        agent = Agent(
            interface = default_agent_interface,
            use_default = True
        )
        controller = RLController()
        controller.setup(
            environment = env,
            agent = agent
        )
        controller.train(
            n_epoch = 1,
            n_env_step = 1,
            n_gradient_step = 1,
            max_dataset_size = 1000,
            batch_size = 10,
            shuffle = False
        )


@pytest.mark.L1
class TestGoalConditionedRLController():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        controller = GoalConditionedRLController()
        assert controller.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        env = GoalReachingTaskEnvironment()
        agent = GoalConditionedAgent()
        controller = GoalConditionedRLController(
            environment = env,
            agent = agent
        )
        assert controller.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        env = GoalReachingTaskEnvironment()
        agent = GoalConditionedAgent()
        controller = GoalConditionedRLController()
        controller.setup(
            environment = env,
            agent = agent
        )
        assert controller.is_available == True

    @pytest.mark.unit
    @pytest.mark.integration
    def test_train_method_should_work(self):
        env = GoalReachingTaskEnvironment(
            configuration = default_environment_configuration
        )
        agent = GoalConditionedAgent(
            interface = default_goal_conditioned_agent_interface,
            use_default = True
        )
        controller = GoalConditionedRLController()
        controller.setup(
            environment = env,
            agent = agent
        )
        controller.train(
            n_epoch = 1,
            n_env_step = 1,
            n_gradient_step = 1,
            max_dataset_size = 1000,
            batch_size = 10,
            shuffle = False
        )
