from curses.ascii import SP
import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.network import PolicyNetwork
from src.network import DiscretePolicyNetwork
from src.network import ContinuousPolicyNetwork
from src.optimizer import MeasureOptimizer
from src.policy import DiscretePolicy
from src.policy import ContinuousPolicy


optimizer_factory = torch.optim.Adam
default_agent_interface_with_discrete_action = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)
default_agent_interface_with_continuous_action = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.CONTINUOUS
)

@pytest.mark.L4
class TestDiscretePolicy():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy = DiscretePolicy()
        assert policy.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = default_agent_interface_with_discrete_action
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = MeasureOptimizer(optimizer_factory)
        policy = DiscretePolicy(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = default_agent_interface_with_discrete_action
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = MeasureOptimizer(optimizer_factory)
        policy = DiscretePolicy()
        policy.setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        policy = DiscretePolicy(
            interface = interface,
            use_default = True
        )
        assert policy.is_available == True
    
    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            policy = DiscretePolicy(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = MeasureOptimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            policy = DiscretePolicy(
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_have_discrete_policy_network(self):
        interface = default_agent_interface_with_discrete_action
        policy = DiscretePolicy(
            interface = interface,
            use_default = True
        )
        assert type(policy.policy_network) is DiscretePolicyNetwork


@pytest.mark.L4
class TestContinuousPolicy():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy = ContinuousPolicy()
        assert policy.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = default_agent_interface_with_continuous_action
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = MeasureOptimizer(optimizer_factory)
        policy = ContinuousPolicy(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = default_agent_interface_with_continuous_action
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = MeasureOptimizer(optimizer_factory)
        policy = ContinuousPolicy()
        policy.setup(
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        assert policy.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_continuous_action
        policy = ContinuousPolicy(
            interface = interface,
            use_default = True
        )
        assert policy.is_available == True
    
    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            policy = ContinuousPolicy(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_continuous_action
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        policy_optimizer = MeasureOptimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            policy = ContinuousPolicy(
                policy_network = policy_network,
                policy_optimizer = policy_optimizer,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_have_continuous_policy_network(self):
        interface = default_agent_interface_with_continuous_action
        policy = ContinuousPolicy(
            interface = interface,
            use_default = True
        )
        assert type(policy.policy_network) is ContinuousPolicyNetwork
