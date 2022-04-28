import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.network import PolicyNetwork
from src.policy import Policy


class TestPolicy():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy = Policy()
        assert policy.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        policy_network = PolicyNetwork()
        policy = Policy(policy_network)
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        policy_network = PolicyNetwork()
        policy = Policy()
        policy.setup(policy_network)
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        policy = Policy(use_default = True)
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        policy_network = PolicyNetwork()
        with pytest.raises(ValueError) as message:
            policy = Policy(
                policy_network = policy_network,
                use_default = True
            )
