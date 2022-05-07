import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import Interface
from src.network import PolicyNetwork
from src.optimizer import Optimizer
from src.policy import Policy


optimizer_factory = torch.optim.Adam

@pytest.mark.L4
class TestPolicy():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy = Policy()
        assert policy.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = Interface(din = 0, dout = 0)
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = Optimizer(optimizer_factory)
        policy = Policy(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = Interface(din = 0, dout = 0)
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = Optimizer(optimizer_factory)
        policy = Policy()
        policy.setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = Interface(din = 0, dout = 0)
        policy = Policy(
            interface = interface,
            use_default = True
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = Interface(din = 0, dout = 0)
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = Optimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            policy = Policy(
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                use_default = True
            )
