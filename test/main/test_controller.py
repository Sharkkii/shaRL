import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.environment import Environment
from src.agent import Agent
from src.controller import Controller


class TestController():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        controller = Controller()
        assert controller.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        env = Environment()
        agent = Agent()
        controller = Controller(env, agent)
        assert controller.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        env = Environment()
        agent = Agent()
        controller = Controller()
        controller.setup(env, agent)
        assert controller.is_available == True
