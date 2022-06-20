import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.common import AgentInterface
from src.const import SpaceType
from src.environment import CartPoleEnvironment
from src.environment import DiscreteMountainCarEnvironment
from src.core import DQNAgent
from src.core import DoubleDQNAgent
from src.core import GCSLAgent
from src.controller import RLController
from src.controller import GoalConditionedRLController

cartpole_agent_interface = AgentInterface(
    sin = (4,),
    sout = (2,),
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)
discrete_mountaincar_agent_interface = AgentInterface(
    sin = (2,),
    sout = (3,),
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)
discrete_mountaincar_goal_conditioned_agent_interface = AgentInterface(
    sin = (2 * 2,),
    sout = (3,),
    tin = SpaceType.CONTINUOUS,
    tout = SpaceType.DISCRETE
)

@pytest.mark.L2
class TestDQN():

    def rl_with_dqn_agent_on_cartpole_env(
        self,
        configuration_env,
        configuration_agent,
        configuration_ctrl
    ):
        # environment
        version = configuration_env["version"]
        
        env = CartPoleEnvironment(
            version = version
        )

        # agent
        eps = configuration_agent["eps"]
        tau = configuration_agent["tau"]
        gamma = configuration_agent["gamma"]

        agent = DQNAgent(
            eps = eps,
            tau = tau,
            gamma = gamma,
            interface = cartpole_agent_interface,
            use_default = True,
        ) 

        # controller
        n_epoch = configuration_ctrl["n_epoch"]
        n_env_step = configuration_ctrl["n_env_step"]
        n_gradient_step = configuration_ctrl["n_gradient_step"]
        max_dataset_size = configuration_ctrl["max_dataset_size"]
        batch_size = configuration_ctrl["batch_size"]

        controller = RLController(
            environment = env,
            agent = agent,
            config_e = None,
            config_a = None
        )
        controller.train(
            n_epoch = n_epoch,
            n_env_step = n_env_step,
            n_gradient_step = n_gradient_step,
            max_dataset_size = max_dataset_size,
            batch_size = batch_size,
            n_train_eval = 5,
            n_test_eval = 5,
            shuffle = True
        )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "version",
        [ "v0", "v1" ]
    )
    def test_dqn_agent_should_work_on_cartpole_env(self, version):
        
        configuration_env = {
            "version": version
        }
        configuration_agent = {
            "eps": 0.20,
            "tau": 0.01,
            "gamma": 0.90
        }
        configuration_ctrl = {
            "n_epoch": 10,
            "n_env_step": 1,
            "n_gradient_step": 1,
            "max_dataset_size": 1000,
            "batch_size": 10,
        }
        self.rl_with_dqn_agent_on_cartpole_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )
        
    @pytest.mark.evaluation
    def test_dqn_agent_should_perform_well_on_cartpole_v0_env(self):

        configuration_env = {
            "version": "v0"
        }
        configuration_agent = {
            "eps": 0.50,
            "tau": 0.01,
            "gamma": 0.90
        }
        configuration_ctrl = {
            "n_epoch": 200,
            "n_env_step": 5,
            "n_gradient_step": 5,
            "max_dataset_size": 20000,
            "batch_size": 100,
        }
        self.rl_with_dqn_agent_on_cartpole_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )
    
    @pytest.mark.evaluation
    def test_dqn_agent_should_perform_well_on_cartpole_v1_env(self):

        configuration_env = {
            "version": "v1"
        }
        configuration_agent = {
            "eps": 0.50,
            "tau": 0.01,
            "gamma": 0.90
        }
        configuration_ctrl = {
            "n_epoch": 500,
            "n_env_step": 10,
            "n_gradient_step": 10,
            "max_dataset_size": 50000,
            "batch_size": 100,
        }
        self.rl_with_dqn_agent_on_cartpole_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )


@pytest.mark.L2
class TestDoubleDQN():

    def rl_with_double_dqn_agent_on_cartpole_env(
        self,
        configuration_env,
        configuration_agent,
        configuration_ctrl
    ):
        # environment
        version = configuration_env["version"]
        
        env = CartPoleEnvironment(
            version = version
        )

        # agent
        eps = configuration_agent["eps"]
        tau = configuration_agent["tau"]
        gamma = configuration_agent["gamma"]

        agent = DoubleDQNAgent(
            eps = eps,
            tau = tau,
            gamma = gamma,
            interface = cartpole_agent_interface,
            use_default = True,
        ) 

        # controller
        n_epoch = configuration_ctrl["n_epoch"]
        n_env_step = configuration_ctrl["n_env_step"]
        n_gradient_step = configuration_ctrl["n_gradient_step"]
        max_dataset_size = configuration_ctrl["max_dataset_size"]
        batch_size = configuration_ctrl["batch_size"]

        controller = RLController(
            environment = env,
            agent = agent,
            config_e = None,
            config_a = None
        )
        controller.train(
            n_epoch = n_epoch,
            n_env_step = n_env_step,
            n_gradient_step = n_gradient_step,
            max_dataset_size = max_dataset_size,
            batch_size = batch_size,
            n_train_eval = 5,
            n_test_eval = 5,
            shuffle = True
        )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "version",
        [ "v0", "v1" ]
    )
    def test_double_dqn_agent_should_work_on_cartpole_env(self, version):
        
        configuration_env = {
            "version": version
        }
        configuration_agent = {
            "eps": 0.20,
            "tau": 0.01,
            "gamma": 0.90
        }
        configuration_ctrl = {
            "n_epoch": 10,
            "n_env_step": 1,
            "n_gradient_step": 1,
            "max_dataset_size": 1000,
            "batch_size": 10,
        }
        self.rl_with_double_dqn_agent_on_cartpole_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )
        
    @pytest.mark.evaluation
    def test_double_dqn_agent_should_perform_well_on_cartpole_v0_env(self):

        configuration_env = {
            "version": "v0"
        }
        configuration_agent = {
            "eps": 0.50,
            "tau": 0.01,
            "gamma": 0.90
        }
        configuration_ctrl = {
            "n_epoch": 200,
            "n_env_step": 5,
            "n_gradient_step": 5,
            "max_dataset_size": 20000,
            "batch_size": 100,
        }
        self.rl_with_double_dqn_agent_on_cartpole_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )
    
    @pytest.mark.evaluation
    def test_double_dqn_agent_should_perform_well_on_cartpole_v1_env(self):

        configuration_env = {
            "version": "v1"
        }
        configuration_agent = {
            "eps": 0.50,
            "tau": 0.01,
            "gamma": 0.90
        }
        configuration_ctrl = {
            "n_epoch": 500,
            "n_env_step": 10,
            "n_gradient_step": 10,
            "max_dataset_size": 50000,
            "batch_size": 100,
        }
        self.rl_with_double_dqn_agent_on_cartpole_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )


class TestGCSL:

    def rl_with_gcsl_agent_on_discrete_mountaincar_env(
        self,
        configuration_env,
        configuration_agent,
        configuration_ctrl
    ):
        # environment
        reward_spec = configuration_env["reward_spec"]
        difficulty = configuration_env["difficulty"]
        
        env = DiscreteMountainCarEnvironment(
            reward_spec = reward_spec,
            difficulty = difficulty
        )

        # agent
        agent = GCSLAgent(
            interface = discrete_mountaincar_goal_conditioned_agent_interface,
            use_default = True
        )

        # controller
        n_epoch = configuration_ctrl["n_epoch"]
        n_env_step = configuration_ctrl["n_env_step"]
        n_gradient_step = configuration_ctrl["n_gradient_step"]
        max_dataset_size = configuration_ctrl["max_dataset_size"]
        batch_size = configuration_ctrl["batch_size"]

        controller = GoalConditionedRLController(
            environment = env,
            agent = agent,
            config_e = None,
            config_a = None
        )
        controller.train(
            n_epoch = n_epoch,
            n_env_step = n_env_step,
            n_gradient_step = n_gradient_step,
            max_dataset_size = max_dataset_size,
            batch_size = batch_size,
            n_train_eval = 5,
            n_test_eval = 5,
            shuffle = True
        )
        
    @pytest.mark.integration
    def test_gcsl_agent_should_work_on_discrete_mountaincar_env(self):

        configuration_env = {
            "reward_spec": "sparse",
            "difficulty": "easy"
        }
        configuration_agent = {}
        configuration_ctrl = {
            "n_epoch": 10,
            "n_env_step": 1,
            "n_gradient_step": 1,
            "max_dataset_size": 1000,
            "batch_size": 10,
        }
        self.rl_with_gcsl_agent_on_discrete_mountaincar_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )

    @pytest.mark.evaluation
    def test_gcsl_agent_should_perform_well_on_discrete_mountaincar_env(self):

        configuration_env = {
            "reward_spec": "sparse",
            "difficulty": "normal"
        }
        configuration_agent = {}
        configuration_ctrl = {
            "n_epoch": 200,
            "n_env_step": 10,
            "n_gradient_step": 1,
            "max_dataset_size": 1000000,
            "batch_size": 1000,
        }
        self.rl_with_gcsl_agent_on_discrete_mountaincar_env(
            configuration_env = configuration_env,
            configuration_agent = configuration_agent,
            configuration_ctrl = configuration_ctrl
        )
