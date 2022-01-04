#### Agent ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from ..actor import Actor
from ..critic import Critic
from ..environment import Model
from ..memory import RLMemory
from ..controller import Phases


class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        actor = None,
        critic = None,
        model = None,
        memory = None,
        gamma = 1.0
    ):
        self.actor = Actor() if (actor is None) else actor
        self.critic = Critic() if (critic is None) else critic
        self.env = None
        self.model = Model() if (model is None) else model
        self.memory = RLMemory() if (memory is None) else memory
        self.gamma = gamma
    
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
        env,
        policy_network = None,
        value_network = None,
        qvalue_network = None,
        policy_optimizer = None,
        value_optimizer = None,
        qvalue_optimizer = None
    ):
        self.env = env
        self.actor.setup(
            env = env,
            policy_network = policy_network,
            policy_optimizer = policy_optimizer
        )
        self.critic.setup(
            env = env,
            value_network = value_network,
            qvalue_network = qvalue_network,
            value_optimizer = value_optimizer,
            qvalue_optimizer = qvalue_optimizer,
        )
        self.actor.setup_with_critic(
            critic = self.critic
        )
        self.critic.setup_with_actor(
            actor = self.actor
        )
        self.model.setup(env)
        self.memory.setup()
    
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
    def interact_with(
        self,
        env
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
        model = None,
        memory = None,
        gamma = 1.0
    ):
        super().__init__(
            actor = actor,
            critic = critic,
            model = model,
            memory = memory,
            gamma = gamma
        )
    
    def reset(
        self
    ):
        super().reset()

    def setup(
        self,
        env,
        policy_network = None,
        value_network = None,
        qvalue_network = None,
        policy_optimizer = None,
        value_optimizer = None,
        qvalue_optimizer = None
    ):
        super().setup(
            env = env,
            policy_network = policy_network,
            value_network = value_network,
            qvalue_network = qvalue_network,
            policy_optimizer = policy_optimizer,
            value_optimizer = value_optimizer,
            qvalue_optimizer = qvalue_optimizer
        )
    
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

    def interact_with(
        self,
        env,
        n_times = 1,
        n_limit = 1000,
        phase = Phases.NONE,
        verbose = False
    ):

        history = []
        for _ in range(n_times):

            t = 0
            state = env.reset()
            done = False
            for _ in range(n_limit):
                if (done): break
                action = self.actor.choose_action(
                    state = state,
                    phase = phase
                )
                next_state, reward, done, info = env.step(action)
                history.append((state, action, reward, next_state))
                state = next_state
                t = t + 1
        
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

    
