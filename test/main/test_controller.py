import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.environment import Environment
from src.agent import Agent
from src.controller import Controller


default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

@pytest.mark.L1
class TestController():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        controller = Controller()
        assert controller.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        env = Environment()
        agent = Agent()
        controller = Controller(
            environment = env,
            agent = agent
        )
        assert controller.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        env = Environment()
        agent = Agent()
        controller = Controller()
        controller.setup(
            environment = env,
            agent = agent
        )
        assert controller.is_available == True

    @pytest.mark.unit
    @pytest.mark.integration
    def test_train_method_should_work(self):
        env = Environment()
        agent = Agent(
            interface = default_agent_interface,
            use_default = True
        )
        controller = Controller()
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
