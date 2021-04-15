#### Agent ####

import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import torch

sys.path.append("../src/")
from utils.utils_env import unzip_trajectory, zip_trajectory
from utils.utils_value import map_over_trajectory


# base
class Agent(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        actor,
        critic,
        memory,
        model,
        gamma=1.0
    ):
        self.env = None
        self.model = model
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.gamma = gamma
    
    @abstractmethod
    def reset(
        self,
    ):
        self.actor.reset()
        self.critic.reset()
        self.memory.reset()
        self.model.reset()

    @abstractmethod
    def setup(
        self,
        env
    ):
        self.env = env
        self.model.setup(env)
    
    @abstractmethod
    def setup_every_epoch(
        epoch,
        n_epoch
    ):
        raise NotImplementedError

    def update_actor(
        self,
        trajs=[],
        n_times=1
    ):
        return self.actor.update(trajs, n_times=n_times)
    
    def update_critic(
        self,
        trajs=[],
        n_times=1
    ):
        return self.critic.update(trajs, n_times=n_times)

    def update_model(
        self,
        n_times=1
    ):
        return self.model.update(trajs, n_times=n_times)

    @abstractmethod
    def update_every_epoch(
        epoch,
        n_epoch
    ):
        raise NotImplementedError

    def interact_with_env(
        self,
        env,
        is_discrete_action_space=True,
        is_deterministic_policy=True,
        n_times=1
    ):
        return self.actor.interact(
            env,
            is_discrete_action_space=is_discrete_action_space,
            is_deterministic_policy=is_deterministic_policy,
            n_times=n_times
        )

    def interact_with_model(
        self,
        is_discrete_action_space=True,
        is_deterministic_policy=True,
        n_times=1
    ):
        return self.actor.interact(
            self.model,
            is_discrete_action_space=is_discrete_action_space,
            is_deterministic_policy=is_deterministic_policy,
            n_times=n_times
        )
    
    def save_trajs(
        self,
        trajs
    ):
        for traj in trajs:
            self.memory.add(traj)

    def load_trajs(
        self,
        n_sample=1
    ):
        traj = self.memory.replay(N=n_sample)
        trajs = [traj]
        return trajs

    def evaluate(
        self,
        env,
        n_eval=1
    ):
        trajs = self.interact_with_env(env, n_times=n_eval)
        durations = []
        cumulative_rewards = []
        for traj in trajs:
            # duration
            duration = len(traj)
            durations.append(duration)
            # cumulative reward
            _, _, reward_traj, _ = unzip_trajectory(traj)
            gamma_traj = np.geomspace(1.0, self.gamma**(duration-1), duration)
            cumulative_reward = np.dot(reward_traj, gamma_traj)
            cumulative_rewards.append(cumulative_reward)

        duration = np.mean(durations)
        cumulative_reward = np.mean(cumulative_rewards)

        # print("duration: ", duration, end=" | ")
        # print("cumulative reward: ", cumulative_reward)
        return duration, cumulative_reward


# default
class DefaultAgent(Agent):

    def __init__(
        self,
        actor,
        critic,
        memory,
        model,
        gamma=1.0
    ):
        self.env = None
        self.model = model
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.gamma = gamma

    def setup(
        self,
        env
    ):
        super().setup(env)

    def setup_every_epoch(epoch, n_epoch):
        pass

    def reset(
        self,
    ):
        super().reset()

    def update_every_epoch(epoch, n_epoch):
        pass


# Q-learning
class QlearningAgent(Agent):

    def __init__(
        self,
        actor,
        critic,
        memory,
        model,
        gamma=1.0
    ):
        super().__init__(
            model=model,
            actor=actor,
            critic=critic,
            memory=memory,
            gamma=gamma
        )

    def setup(
        self,
        env
    ):
        super().setup(env)

    def reset(
        self,
    ):
        super().reset()

    def setup_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        uni = 1.0 / self.model.action_space.n
        eps =  (1 - epoch/n_epoch) * uni
        self.actor.setup(eps=eps)
    
    def update_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        pass


# DQN
DqnAgent = QlearningAgent


# SAC-discrete
class SacAgent(Agent):

    def __init__(
        self,
        actor,
        critic,
        memory,
        model,
        gamma=1.0,
        alpha=1.0
    ):
        super().__init__(
            model=model,
            actor=actor,
            critic=critic,
            memory=memory,
            gamma=gamma
        )
        self.alpha = alpha

    def setup(
        self,
        env
    ):
        super().setup(env)

    def reset(
        self,
    ):
        super().reset()

    def setup_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        pass

    def update_actor(
        self,
        trajs=[],
        n_times=1
    ):
        return self.actor.update(
            trajs,
            value=self.critic.value,
            qvalue=self.critic.qvalue,
            alpha=self.alpha,
            n_times=n_times
        )
    
    def update_critic(
        self,
        trajs=[],
        n_times=1
    ):
        return self.critic.update(
            trajs,
            policy=self.actor.policy,
            alpha=self.alpha,
            n_times=n_times
        )
    
    def update_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        pass

