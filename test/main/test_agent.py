import pytest
import sys, os
import gym
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import SpaceType
from src.common import AgentInterface
from src.common.helper import is_list_of_data
from src.actor import Actor
from src.actor import DiscreteControlActor
from src.actor import ContinuousControlActor
from src.critic import Critic
from src.critic import DiscreteControlCritic
from src.critic import ContinuousControlCritic
from src.agent import Agent
from src.agent import DiscreteControlAgent
from src.agent import ContinuousControlAgent
from src.agent import GoalConditionedAgent
from src.environment import Environment
from src.environment import GoalReachingTaskEnvironment


default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

default_goal_conditioned_agent_interface = AgentInterface(
    sin = 1 * 2,
    sout = 1,
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

@pytest.mark.L2
class TestAgent():

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent",
        [ Agent, DiscreteControlAgent, ContinuousControlAgent ]
    )
    def test_should_be_unavailable_on_empty_initialization(self, TAgent):
        agent = TAgent()
        assert agent.is_available == False

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent, TActor, TCritic",
        [
            (Agent, Actor, Critic),
            (DiscreteControlAgent, DiscreteControlActor, DiscreteControlCritic),(ContinuousControlAgent, ContinuousControlActor, ContinuousControlCritic)
        ]
    )
    def test_should_be_available_on_nonempty_initialization(self, TAgent, TActor, TCritic):
        actor = TActor()
        critic = TCritic()
        agent = TAgent(
            actor = actor,
            critic = critic
        )
        assert agent.is_available == True

    # @pytest.mark.unit
    # @pytest.mark.parametrize(
    #     "TAgent, TActor, TCritic",
    #     [
    #         (Agent, Actor, Critic),
    #         (DiscreteControlAgent, DiscreteControlActor, DiscreteControlCritic),(ContinuousControlAgent, ContinuousControlActor, ContinuousControlCritic)
    #     ]
    # )
    # def test_should_be_available_after_setup(self, TAgent, TActor, TCritic):
    #     actor = TActor()
    #     critic = TCritic()
    #     agent = TAgent()
    #     agent.setup(
    #         actor = actor,
    #         critic = critic
    #     )
    #     assert agent.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent",
        [ Agent, DiscreteControlAgent, ContinuousControlAgent ]
    )
    def test_should_be_available_on_empty_initialization_with_use_default_true(self, TAgent):
        interface = default_agent_interface
        agent = TAgent(
            interface = interface,
            use_default = True
        )
        assert agent.is_available == True
    
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent",
        [ Agent, DiscreteControlAgent, ContinuousControlAgent ]
    )
    def test_should_raise_value_error_with_use_default_true_but_no_interface_specified(self, TAgent):
        with pytest.raises(ValueError) as message:
            agent = TAgent(
                interface = None,
                use_default = True
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent, TActor, TCritic",
        [
            (Agent, Actor, Critic),
            (DiscreteControlAgent, DiscreteControlActor, DiscreteControlCritic),(ContinuousControlAgent, ContinuousControlActor, ContinuousControlCritic)
        ]
    )
    def test_should_raise_value_error_on_nonempty_initialization_with_use_default_true(self, TAgent, TActor, TCritic):
        actor = TActor()
        critic = TCritic()
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            agent = TAgent(
                actor = actor,
                critic = critic,
                interface = interface,
                use_default = True
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent, TActor, TCritic",
        [
            (Agent, Actor, Critic),
            (DiscreteControlAgent, DiscreteControlActor, DiscreteControlCritic),(ContinuousControlAgent, ContinuousControlActor, ContinuousControlCritic)
        ]
    )
    def test_should_accept_dictionary_type_configuration(self, TAgent, TActor, TCritic):
        actor = TActor()
        critic = TCritic()
        interface = default_agent_interface
        configuration = { "key": "value" }
        agent = TAgent(
            actor = actor,
            critic = critic,
            interface = interface,
            configuration = configuration
        )
        assert agent.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TAgent, TActor, TCritic",
        [
            (Agent, Actor, Critic),
            (DiscreteControlAgent, DiscreteControlActor, DiscreteControlCritic),(ContinuousControlAgent, ContinuousControlActor, ContinuousControlCritic)
        ]
    )
    def test_should_reject_non_dictionary_type_configuration(self, TAgent, TActor, TCritic):
        actor = TActor()
        critic = TCritic()
        interface = default_agent_interface
        configuration = ( "value1", "value2", "value3" )
        with pytest.raises(ValueError) as message:
            agent = TAgent(
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
        
        env = Environment(
            configuration = {
                "observation_space": env_observation_space,
                "action_space": env_action_space
            }
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
        
        env = Environment(
            configuration = {
                "observation_space": env_observation_space,
                "action_space": env_action_space
            }
        )

        agent = Agent(
            interface = agent_interface,
            use_default = True
        )
        
        state = env.reset()
        action = agent.choose_action(state)
        assert env.can_accept_action(action = action) == False

    @pytest.mark.unit
    @pytest.mark.integration
    def test_can_interact_with_environment_for_one_step(self):
        
        env = Environment(
            configuration = {
                "observation_space": gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                "action_space": gym.spaces.Discrete(2)
            }
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

        env = Environment(
            configuration = {
                "observation_space": gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                "action_space": gym.spaces.Discrete(2)
            }
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


@pytest.mark.L2
class TestGoalConditionedAgent:

    @pytest.mark.unit
    @pytest.mark.integration
    def test_can_interact_with_environment_for_one_step(self):
        
        env = GoalReachingTaskEnvironment(
            configuration = {
                "observation_space": gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                "action_space": gym.spaces.Discrete(2),
                "goal_space": gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,))
            }
        )

        agent = GoalConditionedAgent(
            interface = default_goal_conditioned_agent_interface,
            use_default = True
        )

        state, goal = env.reset(use_goal = True)
        action = agent.choose_action(
            state = state,
            goal = goal
        )
        assert env.can_accept_action(action) == True
        state, goal, done, info = env.step(action)

    @pytest.mark.unit
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "max_nstep",
        [
            1, 10, 100, 1000
        ]
    )
    def test_can_interact_with_environment(self, max_nstep):

        env = GoalReachingTaskEnvironment(
            configuration = {
                "observation_space": gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,)),
                "action_space": gym.spaces.Discrete(2),
                "goal_space": gym.spaces.Box(low = 0.0, high = 1.0, shape = (1,))
            }
        )

        agent = GoalConditionedAgent(
            interface = default_goal_conditioned_agent_interface,
            use_default = True
        )
        history = agent.interact_with_env(
            env = env,
            max_nstep = max_nstep
        )
        assert is_list_of_data(history) == True
