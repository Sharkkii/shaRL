import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.agent import Agent


class TestAgent():

    @pytest.mark.unit
    def test_init(self):
        agent = Agent()
        assert agent.is_available == False
