#### Agent ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from ..const import PhaseType
from ..common import AgentInterface
from ..common import Component
from ..common import SARS
from ..actor import Actor
from ..critic import Critic
from ..environment import Model
from ..memory import RLMemory


class BaseAgent(Component, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        actor = None,
        critic = None,
        interface = None,
        configuration = None,
        model = None, # will be in `component`
        memory = None, # will be in `component`
        gamma = 1.0, # will be in `configuration`
        use_default = False
    ):
        Component.__init__(self)
        if (use_default):
            if (not ((actor is None) and (critic is None))):
                raise ValueError("`actor` & `critic` must be None if `use_default = True`")
            if (type(interface) is not AgentInterface):
                raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`")
            actor = Actor(
                interface = interface,
                use_default = True
            )
            critic = Critic(
                interface = interface,
                use_default = True
            )

        if ((configuration is not None) and (type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None")

        self.actor = Actor() if (actor is None) else actor
        self.critic = Critic() if (critic is None) else critic
        self.env = None
        self.model = Model() if (model is None) else model
        self.memory = RLMemory() if (memory is None) else memory
        self.gamma = gamma
        self.setup(
            actor = actor,
            critic = critic
        )
    
    @abstractmethod
    def reset(
        self
    ):
        self.actor.reset()
        self.critic.reset()
        self.model.reset()
        self.memory.reset()

    @abstractmethod
    def setup(
        self,
        env = None, # will be deprecated
        policy_network = None, # will be deprecated
        value_network = None, # will be deprecated
        qvalue_network = None, # will be deprecated
        policy_optimizer = None, # will be deprecated
        value_optimizer = None, # will be deprecated
        qvalue_optimizer = None, # will be deprecated
        actor = None,
        critic = None
    ):
        if ((actor is not None) and (critic is not None)):
            self.actor = actor
            self.critic = critic
            self.critic.setup_with_actor(
                actor = self.actor
            )
            self.actor.setup_with_critic(
                critic = self.critic
            )
            self._become_available()

        # self.actor.setup(
        #     env = env,
        #     policy_network = policy_network,
        #     policy_optimizer = policy_optimizer
        # )
        # self.critic.setup(
        #     env = env,
        #     value_network = value_network,
        #     qvalue_network = qvalue_network,
        #     value_optimizer = value_optimizer,
        #     qvalue_optimizer = qvalue_optimizer,
        # )
        # self.model.setup(env)
        # self.memory.setup()
    
    @abstractmethod
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        raise NotImplementedError

    @abstractmethod
    def setup_on_every_step(
        self,
        step,
        n_step
    ):
        raise NotImplementedError

    def choose_action(
        self,
        state,
        information = None
    ):
        return self.actor.choose_action(
            state = state,
            information = information
        )

    def train(
        self
    ):
        self.actor.train()
        self.critic.train()
    
    def eval(
        self
    ):
        self.actor.eval()
        self.critic.eval()

    def save(
        self,
        path_to_policy,
        path_to_value,
        path_to_qvalue
    ):
        self.actor.save(
            path_to_policy = path_to_policy
        )
        self.critic.save(
            path_to_value = path_to_value,
            path_to_qvalue = path_to_qvalue
        )

    def load(
        self,
        path_to_policy,
        path_to_value,
        path_to_qvalue
    ):
        self.actor.load(
            path_to_policy = path_to_policy
        )
        self.critic.load(
            path_to_value = path_to_value,
            path_to_qvalue = path_to_qvalue
        )

    @abstractmethod
    def update_model(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError

    @abstractmethod
    def update_actor(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError
    
    @abstractmethod
    def update_critic(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError

    # @abstractmethod
    # def update_every_epoch(
    #     epoch,
    #     n_epoch
    # ):
    #     raise NotImplementedError

    @abstractmethod
    def interact_with_env(
        self,
        env,
        information = None,
        use_info = False
    ):
        raise NotImplementedError
    
    @abstractmethod
    def save_history(
        self,
        history
    ):
        raise NotImplementedError

    @abstractmethod
    def load_history(
        self,
        n_sample = 0
    ):
        raise NotImplementedError

class Agent(BaseAgent):

    def __init__(
        self,
        actor = None,
        critic = None,
        interface = None,
        configuration = None,
        model = None, # will be in `component`
        memory = None, # will be in `component`
        gamma = 1.0, # will be in `configuration`
        use_default = False
    ):
        super().__init__(
            actor = actor,
            critic = critic,
            interface = interface,
            configuration = configuration,
            model = model,
            memory = memory,
            gamma = gamma,
            use_default = use_default
        )
    
    def reset(
        self
    ):
        super().reset()

    def setup(
        self,
        env = None, # will be deprecated
        policy_network = None, # will be deprecated
        value_network = None, # will be deprecated
        qvalue_network = None, # will be deprecated
        policy_optimizer = None, # will be deprecated
        value_optimizer = None, # will be deprecated
        qvalue_optimizer = None, # will be deprecated
        actor = None,
        critic = None
    ):
        super().setup(
            actor = actor,
            critic = critic
        )

        # super().setup(
        #     env = env,
        #     policy_network = policy_network,
        #     value_network = value_network,
        #     qvalue_network = qvalue_network,
        #     policy_optimizer = policy_optimizer,
        #     value_optimizer = value_optimizer,
        #     qvalue_optimizer = qvalue_optimizer
        # )
    
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        pass

    def setup_on_every_step(
        self,
        step,
        n_step
    ):
        pass

    def update_model(
        self,
        trajectory,
        n_times = 1
    ):
        return self.model.update(
            trajectory,
            n_times = n_times
        )

    def update_actor(
        self,
        trajectory,
        n_times = 1
    ):
        return self.actor.update(
            self.critic,
            trajectory,
            n_times = n_times
        )
    
    def update_critic(
        self,
        trajectory,
        n_times = 1
    ):
        return self.critic.update(
            self.actor,
            trajectory, 
            n_times = n_times
        )

    def interact_with_env(
        self,
        env,
        # n_times = 1, # deprecated -> use `n_episode`
        # n_limit = 1000, # deprecated -> use `max_nstep`
        n_episode = 1,
        max_nstep = 1000,
        information = None,
        use_info = False,
        verbose = False,
    ):

        if (information is None):
            information = {}

        if (type(information) is not dict):
            raise ValueError("`information` must be a 'dictionary' object.")

        # action_space
        information["action_space"] = env.action_space

        history = []
        info_history = []

        for _ in range(n_episode):

            t = 0
            state = env.reset()
            done = False
            for _ in range(max_nstep):
                if (done): break
                action = self.actor.choose_action(
                    state = state,
                    information = information
                )
                next_state, reward, done, info = env.step(action)
                data = SARS(state, action, reward, next_state)
                history.append(data)
                state = next_state
                if (use_info):
                    info_history.append(info)
                t = t + 1
        
        if (use_info):
            return history, info_history
        else:
            return history
    
    def save_history(
        self,
        history
    ):
        self.memory.save(history)

    def load_history(
        self
    ):
        return self.memory.load()
    
    def replay_history(
        self,
        n_sample = 0
    ):
        return self.memory.replay(
            n_sample = n_sample
        )
