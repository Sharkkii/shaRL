import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.critic import Critic


class TestCritic():

    @pytest.mark.unit
    def test_init(self):
        critic = Critic()
        assert critic.is_available == False
