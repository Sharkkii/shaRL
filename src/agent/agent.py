#### Agent ####

import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import torch


class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        actor,
        critic,
        model,
        memory = None,
        gamma = 1.0
    ):
        self.actor = actor
        self.critic = critic
        self.env = None
        self.model = model
        self.memory = memory
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
        env
    ):
        self.actor.setup()
        self.critic.setup()
        self.env = env
        self.model.setup(env)
        self.memory.setup()
    
    # @abstractmethod
    # def setup_every_epoch(
    #     epoch,
    #     n_epoch
    # ):
    #     raise NotImplementedError

    @abstractmethod
    def update_model(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError
        # return self.model.update(trajs, n_times=n_times)

    @abstractmethod
    def update_actor(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError
        # return self.actor.update(trajs, n_times=n_times)
    
    @abstractmethod
    def update_critic(
        self,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError
        # return self.critic.update(trajs, n_times=n_times)

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
        actor,
        critic,
        model,
        memory,
        gamma=1.0
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
        env
    ):
        super().setup(env)
    
    # def setup_every_epoch(
    #     epoch,
    #     n_epoch
    # ):
    #     raise NotImplementedError

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
            trajectory,
            n_times = n_times
        )
    
    def update_critic(
        self,
        trajectory,
        n_times = 1
    ):
        return self.critic.update(
            trajectory, 
            n_times = n_times
        )

    # def update_every_epoch(
    #     epoch,
    #     n_epoch
    # ):
    #     raise NotImplementedError

    def interact_with(
        self,
        env,
        n_times = 1
    ):
        return []
    
    def save_history(
        self,
        history
    ):
        pass

    def load_history(
        self,
        n_sample = 0
    ):
        return []
