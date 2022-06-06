import torch
import gym
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.policy import Policy
from src.policy import DiscretePolicy
from src.policy import ContinuousPolicy
from src.actor import Actor
from src.actor import DiscreteControlActor
from src.actor import ContinuousControlActor
from src.environment import Environment

default_discrete_observation_space = gym.spaces.Discrete(2)
default_continuous_observation_space = gym.spaces.Box(0, 1, shape=(1,))
default_discrete_action_space = gym.spaces.Discrete(2)
default_continuous_action_space = gym.spaces.Box(0, 1, shape=(1,))

default_discrete_action_environment_configuration = {
    "observation_space": default_continuous_observation_space,
    "action_space": default_discrete_action_space
}
default_continuous_action_environment_configuration = {
    "observation_space": default_continuous_observation_space,
    "action_space": default_continuous_action_space
}

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
default_agent_interface = default_agent_interface_with_discrete_action

@pytest.mark.L3
class TestActor():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        actor = Actor()
        assert actor.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        policy = Policy()
        actor = Actor(
            policy = policy
        )
        assert actor.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        policy = Policy()
        actor = Actor()
        actor.setup(
            policy = policy
        )
        assert actor.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface
        actor = Actor(
            interface = interface,
            use_default = True
        )
        assert actor.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            actor = Actor(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        policy = Policy()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            actor = Actor(
                policy = policy,
                interface = interface,
                use_default = True
            )
    
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "interface",
        [
            default_agent_interface_with_discrete_action,
            default_agent_interface_with_continuous_action
        ]
    )
    def test_we_can_check_whether_pointwise_estimation_is_available(self, interface):
        actor = Actor(
            interface = interface,
            use_default = True
        )
        flag = actor.can_pointwise_estimate
        assert type(flag) is bool

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "interface",
        [
            default_agent_interface_with_discrete_action,
            default_agent_interface_with_continuous_action
        ]
    )
    def test_we_can_check_whether_density_estimation_is_available(self, interface):
        actor = Actor(
            interface = interface,
            use_default = True
        )
        flag = actor.can_density_estimate
        assert type(flag) is bool


class TestDiscreteControlActor:

    @pytest.mark.unit
    def test_choose_action_should_return_valid_discrete_action(self):
        interface = default_agent_interface_with_discrete_action
        actor = DiscreteControlActor(
            interface = interface,
            use_default = True
        )

        env = Environment(
            configuration = default_discrete_action_environment_configuration
        )
        state = env.reset()
        action = actor.choose_action(state)
        _ = env.step(action) # check whether valid
        assert type(action) is torch.Tensor
        assert action.dtype is torch.long

    @pytest.mark.unit
    def test_call_should_return_valid_discrete_action(self):
        interface = default_agent_interface_with_discrete_action
        actor = DiscreteControlActor(
            interface = interface,
            use_default = True
        )

        env = Environment(
            configuration = default_discrete_action_environment_configuration
        )
        state = env.reset()
        action = actor(state)
        _ = env.step(action) # check whether valid
        assert type(action) is torch.Tensor
        assert action.dtype is torch.long


class TestContinuousControlActor:

    @pytest.mark.unit
    def test_choose_action_should_return_valid_continuous_action(self):
        interface = default_agent_interface_with_continuous_action
        actor = ContinuousControlActor(
            interface = interface,
            use_default = True
        )

        env = Environment(
            configuration = default_continuous_action_environment_configuration
        )
        state = env.reset()
        action = actor.choose_action(state)
        _ = env.step(action) # check whether valid
        assert type(action) is torch.Tensor
        assert action.dtype is torch.float32

    @pytest.mark.unit
    def test_call_should_return_valid_continuous_action(self):
        interface = default_agent_interface_with_continuous_action
        actor = ContinuousControlActor(
            interface = interface,
            use_default = True
        )

        env = Environment(
            configuration = default_continuous_action_environment_configuration
        )
        state = env.reset()
        action = actor(state)
        _ = env.step(action) # check whether valid
        assert type(action) is torch.Tensor
        assert action.dtype is torch.float32
