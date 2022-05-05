import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.value import Value, QValue
from src.critic import Critic


@pytest.mark.L3
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
        critic.setup(value, qvalue)
        assert critic.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        critic = Critic(use_default = True)
        assert critic.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        value = Value()
        qvalue = QValue()
        with pytest.raises(ValueError) as message:
            critic = Critic(
                value = value,
                qvalue = qvalue,
                use_default = True
            )
