from email.policy import default
import pytest
import sys, os
import gym
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.common.helper import is_list_of_data
from src.actor import Actor
from src.critic import Critic
from src.agent import Agent
from src.environment import Environment


default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

@pytest.mark.L2
class TestAgent():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        agent = Agent()
        assert agent.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        actor = Actor()
        critic = Critic()
        agent = Agent(actor, critic)
        assert agent.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        actor = Actor()
        critic = Critic()
        agent = Agent()
        agent.setup(
            actor = actor,
            critic = critic
        )
        assert agent.is_available == True

    @pytest.mark.unit
    def test_should_be_available_on_empty_initialization_with_use_default_true(self):
        interface = default_agent_interface
        agent = Agent(
            interface = interface,
            use_default = True
        )
        assert agent.is_available == True
    
    @pytest.mark.unit
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self):
        with pytest.raises(ValueError) as message:
            agent = Agent(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self):
        actor = Actor()
        critic = Critic()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            agent = Agent(
                actor = actor,
                critic = critic,
                interface = interface,
                use_default = True
            )

    @pytest.mark.unit
    def test_should_accept_dictionary_type_configuration(self):
        actor = Actor()
        critic = Critic()
        interface = default_agent_interface
        configuration = { "key": "value" }
        agent = Agent(
            actor = actor,
            critic = critic,
            interface = interface,
            configuration = configuration
        )
        assert agent.is_available == True

    @pytest.mark.unit
    def test_should_reject_non_dictionary_type_configuration(self):
        actor = Actor()
        critic = Critic()
        interface = default_agent_interface
        configuration = ( "value1", "value2", "value3" )
        with pytest.raises(ValueError) as message:
            agent = Agent(
                actor = actor,
                critic = critic,
                interface = interface,
                configuration = configuration
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_observation_space, env_action_space, agent_interface",
        [
            (
                gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                gym.spaces.Discrete(2),
                AgentInterface(sin = 1, sout = 1, tin = SpaceType.CONTINUOUS, tout = SpaceType.DISCRETE)
            ),
            (
                gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                AgentInterface(sin = 1, sout = 1, tin = SpaceType.CONTINUOUS, tout = SpaceType.CONTINUOUS)
            ),
            (
                gym.spaces.Box(low = 0.0, high = 1.0, shape = (5,)),
                gym.spaces.Discrete(2),
                AgentInterface(sin = 5, sout = 1, tin = SpaceType.CONTINUOUS, tout = SpaceType.DISCRETE)
            ),
        ]
    )
    def test_action_should_be_accepted_if_valid(
        self,
        env_observation_space,
        env_action_space,
        agent_interface
    ):
        
        env = Environment()
        env.setup(
            observation_space = env_observation_space,
            action_space = env_action_space
        )

        agent = Agent(
            interface = agent_interface,
            use_default = True
        )
        
        state = env.reset()
        action = agent.choose_action(state)
        assert env.can_accept_action(action = action) == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_observation_space, env_action_space, agent_interface",
        [
            # (
                # Discrete(X) -> sout = 1
                # gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                # gym.spaces.Discrete(2),
                # AgentInterface(sin = 1, sout = 10, tin = SpaceType.CONTINUOUS, tout = SpaceType.DISCRETE)
            # ),
            (
                # Box(shape = X) -> sout = X
                gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                AgentInterface(sin = 1, sout = 10, tin = SpaceType.CONTINUOUS, tout = SpaceType.CONTINUOUS)
            )
        ]
    )
    def test_action_should_be_rejected_if_invalid(
        self,
        env_observation_space,
        env_action_space,
        agent_interface
    ):
        
        env = Environment()
        env.setup(
            observation_space = env_observation_space,
            action_space = env_action_space
        )

        agent = Agent(
            interface = agent_interface,
            use_default = True
        )
        
        state = env.reset()
        action = agent.choose_action(state)
        assert env.can_accept_action(action = action) == False

    @pytest.mark.integration
    def test_can_interact_with_environment_for_one_step(self):
        
        env = Environment()
        env.setup(
            observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
            action_space = gym.spaces.Discrete(2)
        )

        agent = Agent(
            interface = default_agent_interface,
            use_default = True
        )

        state = env.reset()
        action = agent.choose_action(state)
        assert env.can_accept_action(action) == True
        state, reward, done, info = env.step(action)

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "max_nstep",
        [
            1, 10, 100, 1000
        ]
    )
    def test_can_interact_with_environment(self, max_nstep):

        env = Environment()
        env.setup(
            observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
            action_space = gym.spaces.Discrete(2)
        )

        agent = Agent(
            interface = default_agent_interface,
            use_default = True
        )
        history = agent.interact_with_env(
            env = env,
            max_nstep = max_nstep
        )
        assert is_list_of_data(history) == True
