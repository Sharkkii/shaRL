import torch
import gym
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.environment import Environment
from src.network import ValueNetwork, QValueNetwork, PolicyNetwork
from src.network import DiscreteQValueNetwork
from src.network import ContinuousQValueNetwork
from src.network import DiscretePolicyNetwork
from src.network import ContinuousPolicyNetwork


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

class CallableObject():
    
    def __init__(self):
        pass

    def __call__(x):
        return x


@pytest.mark.L5
class TestValueNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        value_network = ValueNetwork()
        assert value_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        value_network = ValueNetwork(network)
        assert value_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        assert value_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            value_network = ValueNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface_with_discrete_action
        with pytest.raises(ValueError) as message:
            value_network = ValueNetwork(
                value_network = network,
                interface = interface,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        value_network = ValueNetwork()
        value_network.setup(network)
        assert value_network.is_available == True


@pytest.mark.L5
class TestDiscreteQValueNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue_network = DiscreteQValueNetwork()
        assert qvalue_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        qvalue_network = DiscreteQValueNetwork(network)
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        qvalue_network = DiscreteQValueNetwork()
        qvalue_network.setup(network)
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        qvalue_network = DiscreteQValueNetwork(
            interface = interface,
            use_default = True
        )
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            qvalue_network = DiscreteQValueNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface_with_discrete_action
        with pytest.raises(ValueError) as message:
            qvalue_network = DiscreteQValueNetwork(
                qvalue_network = network,
                interface = interface,
                use_default = True
            )


@pytest.mark.L5
class TestContinuousQValueNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue_network = ContinuousQValueNetwork()
        assert qvalue_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        qvalue_network = ContinuousQValueNetwork(network)
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        qvalue_network = ContinuousQValueNetwork()
        qvalue_network.setup(network)
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_continuous_action
        qvalue_network = ContinuousQValueNetwork(
            interface = interface,
            use_default = True
        )
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            qvalue_network = ContinuousQValueNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface_with_continuous_action
        with pytest.raises(ValueError) as message:
            qvalue_network = ContinuousQValueNetwork(
                qvalue_network = network,
                interface = interface,
                use_default = True
            )

@pytest.mark.L5
class TestDiscretePolicyNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy_network = DiscretePolicyNetwork()
        assert policy_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        policy_network = DiscretePolicyNetwork(network)
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        policy_network = DiscretePolicyNetwork()
        policy_network.setup(network)
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        policy_network = DiscretePolicyNetwork(
            interface = interface,
            use_default = True
        )
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            policy_network = DiscretePolicyNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface_with_discrete_action
        with pytest.raises(ValueError) as message:
            policy_network = DiscretePolicyNetwork(
                policy_network = network,
                interface = interface,
                use_default = True
            )


@pytest.mark.L5
class TestContinuousPolicyNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy_network = ContinuousPolicyNetwork()
        assert policy_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        policy_network = ContinuousPolicyNetwork(network)
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        policy_network = ContinuousPolicyNetwork()
        policy_network.setup(network)
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_continuous_action
        policy_network = ContinuousPolicyNetwork(
            interface = interface,
            use_default = True
        )
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            policy_network = ContinuousPolicyNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface_with_continuous_action
        with pytest.raises(ValueError) as message:
            policy_network = ContinuousPolicyNetwork(
                policy_network = network,
                interface = interface,
                use_default = True
            )
