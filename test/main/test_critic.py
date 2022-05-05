import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.value import Value, QValue
from src.critic import Critic


class TestCritic():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        critic = Critic()
        assert critic.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        value = Value()
        qvalue = QValue()
        critic = Critic(value, qvalue)
        assert critic.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        value = Value()
        qvalue = QValue()
        critic = Critic()
        critic.setup(
            value = value,
            qvalue = qvalue
        )
        assert critic.is_available == True
