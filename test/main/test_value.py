from code import interact
import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import Interface
from src.network import ValueNetwork, QValueNetwork
from src.optimizer import Optimizer
from src.value import Value, QValue


optimizer_factory = torch.optim.Adam

@pytest.mark.L4
class TestValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        value = Value()
        assert value.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = Interface(din = 0, dout = 0)
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        value_optimizer = Optimizer(optimizer_factory)
        value = Value(
            value_network = value_network,
            value_optimizer = value_optimizer
        )
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = Interface(din = 0, dout = 0)
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        value_optimizer = Optimizer(optimizer_factory)
        value = Value()
        value.setup(
            value_network = value_network,
            value_optimizer = value_optimizer
        )
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = Interface(din = 0, dout = 0)
        value = Value(
            interface = interface,
            use_default = True
        )
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            value = Value(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = Interface(din = 0, dout = 0)
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        value_optimizer = Optimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            value = Value(
                value_network = value_network,
                value_optimizer = value_optimizer,
                interface = interface,
                use_default = True
            )

@pytest.mark.L4
class TestQValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue = QValue()
        assert qvalue.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = Interface(din = 0, dout = 0)
        qvalue_network = QValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = Optimizer(optimizer_factory)
        qvalue = QValue(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = Interface(din = 0, dout = 0)
        qvalue_network = QValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = Optimizer(optimizer_factory)
        qvalue = QValue()
        qvalue.setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = Interface(din = 0, dout = 0)
        qvalue = QValue(
            interface = interface,
            use_default = True
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            qvalue = QValue(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = Interface(din = 0, dout = 0)
        qvalue_network = QValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = Optimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            qvalue = QValue(
                qvalue_network = qvalue_network,
                qvalue_optimizer = qvalue_optimizer,
                interface = interface,
                use_default = True
            )
