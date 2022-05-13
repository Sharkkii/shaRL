import torch
import gym
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import MeasureType
from src.const import SpaceType
from src.common import AgentInterface
from src.environment import Environment
from src.network import ValueNetwork, QValueNetwork, PolicyNetwork


default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1
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
        interface = default_agent_interface
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
        interface = default_agent_interface
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
class TestQValueNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue_network = QValueNetwork()
        assert qvalue_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        qvalue_network = QValueNetwork(network)
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        qvalue_network = QValueNetwork()
        qvalue_network.setup(network)
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface
        qvalue_network = QValueNetwork(
            interface = interface,
            use_default = True
        )
        assert qvalue_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            qvalue_network = QValueNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            qvalue_network = QValueNetwork(
                qvalue_network = network,
                interface = interface,
                use_default = True
            )

@pytest.mark.L5
class TestPolicyNetwork():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy_network = PolicyNetwork()
        assert policy_network.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = CallableObject()
        policy_network = PolicyNetwork(network)
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = CallableObject()
        policy_network = PolicyNetwork()
        policy_network.setup(network)
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface
        policy_network = PolicyNetwork(
            interface = interface,
            use_default = True
        )
        assert policy_network.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            policy_network = PolicyNetwork(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        network = CallableObject()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            policy_network = PolicyNetwork(
                policy_network = network,
                interface = interface,
                use_default = True
            )
