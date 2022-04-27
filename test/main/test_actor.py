import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.policy import Policy
from src.actor import Actor


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
        actor.setup(policy)
        assert actor.is_available == True
