import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.network import ValueNetwork, QValueNetwork
from src.value import Value, QValue


class TestValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        value = Value()
        assert value.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        value_network = ValueNetwork()
        value = Value(value_network)
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        value_network = ValueNetwork()
        value = Value()
        value.setup(value_network)
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        value = Value(use_default = True)
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        value_network = ValueNetwork()
        with pytest.raises(ValueError) as message:
            value = Value(
                value_network = value_network,
                use_default = True
            )

class TestQValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue = QValue()
        assert qvalue.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        qvalue_network = QValueNetwork()
        qvalue = QValue(qvalue_network)
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        qvalue_network = QValueNetwork()
        qvalue = QValue()
        qvalue.setup(qvalue_network)
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        qvalue = QValue(use_default = True)
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        qvalue_network = QValueNetwork()
        with pytest.raises(ValueError) as message:
            qvalue = QValue(
                qvalue_network = qvalue_network,
                use_default = True
            )
