import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.value import Value, QValue
from src.value import DiscreteQValue
from src.value import ContinuousQValue
from src.critic import Critic
from src.critic import DiscreteControlCritic
from src.critic import ContinuousControlCritic


default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

@pytest.mark.L3
class TestCritic():

    # @pytest.mark.unit
    # @pytest.mark.parametrize(
    #     "TCritic",
    #     [ Critic, DiscreteControlCritic, ContinuousControlCritic ]
    # )
    # def test_should_be_unavailable_on_empty_initialization(self, TCritic):
    #     critic = TCritic()
    #     assert critic.is_available == False

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TCritic, TValue, TQValue",
        [
            (Critic, Value, QValue),
            (DiscreteControlCritic, Value, DiscreteQValue),
            (ContinuousControlCritic, Value, ContinuousQValue)
        ]
    )
    def test_should_be_available_on_nonempty_initialization(self, TCritic, TValue, TQValue):
        value = TValue()
        qvalue = TQValue()
        critic = TCritic(
            value = value,
            qvalue = qvalue
        )
        assert critic.is_available == True

    # @pytest.mark.unit
    # @pytest.mark.parametrize(
    #     "TCritic, TValue, TQValue",
    #     [
    #         (Critic, Value, QValue),
    #         (DiscreteControlCritic, Value, DiscreteQValue),
    #         (ContinuousControlCritic, Value, ContinuousQValue)
    #     ]
    # )
    # def test_should_be_available_after_setup(self, TCritic, TValue, TQValue):
    #     value = TValue()
    #     qvalue = TQValue()
    #     critic = TCritic()
    #     critic.setup(
    #         value = value,
    #         qvalue = qvalue
    #     )
    #     assert critic.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TCritic",
        [ Critic, DiscreteControlCritic, ContinuousControlCritic ]
    )
    def test_should_be_available_on_empty_initialization_with_use_default_true(self, TCritic):
        interface = default_agent_interface
        critic = TCritic(
            interface = interface,
            use_default = True
        )
        assert critic.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TCritic",
        [ Critic, DiscreteControlCritic, ContinuousControlCritic ]
    )
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self, TCritic):
        with pytest.raises(ValueError) as message:
            critic = TCritic(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TCritic, TValue, TQValue",
        [
            (Critic, Value, QValue),
            (DiscreteControlCritic, Value, DiscreteQValue),
            (ContinuousControlCritic, Value, ContinuousQValue)
        ]
    )
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self, TCritic, TValue, TQValue):
        value = TValue()
        qvalue = TQValue()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            critic = TCritic(
                value = value,
                qvalue = qvalue,
                interface = interface,
                use_default = True
            )

    # @pytest.mark.unit
    # def test_should_have_discrete_qvalue_if_interface_specifies_so(self):
    #     interface = AgentInterface(
    #         sin = 1,
    #         sout = 1,
    #         tin = SpaceType.CONTINUOUS,
    #         tout = SpaceType.DISCRETE
    #     )
    #     critic = Critic(
    #         interface = interface,
    #         use_default = True
    #     )
    #     assert type(critic.value) is Value
    #     assert type(critic.qvalue) is DiscreteQValue 
    
    # @pytest.mark.unit
    # def test_should_have_continuous_qvalue_if_interface_specifies_so(self):
    #     interface = AgentInterface(
    #         sin = 1,
    #         sout = 1,
    #         tin = SpaceType.CONTINUOUS,
    #         tout = SpaceType.CONTINUOUS
    #     )
    #     critic = Critic(
    #         interface = interface,
    #         use_default = True
    #     )
    #     assert type(critic.value) is Value
    #     assert type(critic.qvalue) is ContinuousQValue
