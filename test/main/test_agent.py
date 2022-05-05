import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.actor import Actor
from src.critic import Critic
from src.agent import Agent


class TestAgent():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        agent = Agent()
        assert agent.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        actor = Actor()
        critic = Critic()
        agent = Agent(actor, critic)
        assert agent.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        actor = Actor()
        critic = Critic()
        agent = Agent()
        agent.setup(
            actor = actor,
            critic = critic
        )
        assert agent.is_available == True
