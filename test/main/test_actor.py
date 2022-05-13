import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.common import AgentInterface
from src.policy import Policy
from src.actor import Actor


default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1
)

@pytest.mark.L3
class TestActor():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        actor = Actor()
        assert actor.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        policy = Policy()
        actor = Actor(policy)
        assert actor.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        policy = Policy()
        actor = Actor()
        actor.setup(
            policy = policy
        )
        assert actor.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface
        actor = Actor(
            interface = interface,
            use_default = True
        )
        assert actor.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            actor = Actor(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        policy = Policy()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            actor = Actor(
                policy = policy,
                interface = interface,
                use_default = True
            )
