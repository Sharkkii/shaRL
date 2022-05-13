from code import interact
import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.network import ValueNetwork, QValueNetwork
from src.network import DiscreteQValueNetwork
from src.network import ContinuousQValueNetwork
from src.optimizer import MeasureOptimizer
from src.value import Value, QValue
from src.value import DiscreteQValue
from src.value import ContinuousQValue


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
class TestValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        value = Value()
        assert value.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = default_agent_interface_with_discrete_action
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        value_optimizer = MeasureOptimizer(optimizer_factory)
        value = Value(
            value_network = value_network,
            value_optimizer = value_optimizer
        )
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = default_agent_interface_with_discrete_action
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        value_optimizer = MeasureOptimizer(optimizer_factory)
        value = Value()
        value.setup(
            value_network = value_network,
            value_optimizer = value_optimizer
        )
        assert value.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
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
        interface = default_agent_interface_with_discrete_action
        value_network = ValueNetwork(
            interface = interface,
            use_default = True
        )
        value_optimizer = MeasureOptimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            value = Value(
                value_network = value_network,
                value_optimizer = value_optimizer,
                interface = interface,
                use_default = True
            )

@pytest.mark.L4
class TestDiscreteQValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue = DiscreteQValue()
        assert qvalue.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = default_agent_interface_with_discrete_action
        qvalue_network = DiscreteQValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = MeasureOptimizer(optimizer_factory)
        qvalue = DiscreteQValue(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = default_agent_interface_with_discrete_action
        qvalue_network = DiscreteQValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = MeasureOptimizer(optimizer_factory)
        qvalue = DiscreteQValue()
        qvalue.setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        qvalue = DiscreteQValue(
            interface = interface,
            use_default = True
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            qvalue = DiscreteQValue(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_discrete_action
        qvalue_network = DiscreteQValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = MeasureOptimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            qvalue = DiscreteQValue(
                qvalue_network = qvalue_network,
                qvalue_optimizer = qvalue_optimizer,
                interface = interface,
                use_default = True
            )


@pytest.mark.L4
class TestContinuousQValue():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        qvalue = ContinuousQValue()
        assert qvalue.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        interface = default_agent_interface_with_continuous_action
        qvalue_network = ContinuousQValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = MeasureOptimizer(optimizer_factory)
        qvalue = ContinuousQValue(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        interface = default_agent_interface_with_continuous_action
        qvalue_network = ContinuousQValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = MeasureOptimizer(optimizer_factory)
        qvalue = ContinuousQValue()
        qvalue.setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_continuous_action
        qvalue = ContinuousQValue(
            interface = interface,
            use_default = True
        )
        assert qvalue.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            qvalue = ContinuousQValue(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        interface = default_agent_interface_with_continuous_action
        qvalue_network = ContinuousQValueNetwork(
            interface = interface,
            use_default = True
        )
        qvalue_optimizer = MeasureOptimizer(optimizer_factory)
        with pytest.raises(ValueError) as message:
            qvalue = ContinuousQValue(
                qvalue_network = qvalue_network,
                qvalue_optimizer = qvalue_optimizer,
                interface = interface,
                use_default = True
            )
