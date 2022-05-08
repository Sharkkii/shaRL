import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.common import AgentInterface
from src.actor import Actor
from src.critic import Critic
from src.agent import Agent


@pytest.mark.L2
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

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = AgentInterface(din = 0, dout = 0)
        agent = Agent(
            interface = interface,
            use_default = True
        )
        assert agent.is_available == True
    
    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            agent = Agent(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        actor = Actor()
        critic = Critic()
        interface = AgentInterface(din = 0, dout = 0)
        with pytest.raises(ValueError) as message:
            agent = Agent(
                actor = actor,
                critic = critic,
                interface = interface,
                use_default = True
            )
