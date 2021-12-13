#### Soft Actor Critic ####

import numpy as np
import torch
import torch.nn.functional as F
import gym

from ..value import Value, ContinuousQValue
from ..policy import ContinuousPolicy
from ..actor import Actor
from ..critic import Critic
from ..agent import Agent
from ..controller import Phases


class SoftActorCritic(Agent):

    def __init__(
        self,
        actor = None,
        critic = None,
        model = None,
        memory = None,
        gamma = 1.0,
        alpha = 1.0,
        alpha_decay = 1.0,
        tau = 0.5
    ):
        actor = SACActor(
            alpha = alpha,
            alpha_decay = alpha_decay,
            tau = tau
        )
        critic = SACCritic(
            gamma = gamma,
            alpha = alpha,
            tau = tau
        )
        super().__init__(
            actor = actor,
            critic = critic,
            model = model,
            memory = memory,
            gamma = gamma
        )

class SACActor(Actor):

    def __init__(
        self,
        policy = None,
        alpha = 1.0,
        alpha_decay = 1.0,
        tau = 0.5
    ):
        assert(alpha >= 0.0)
        assert(0.0 < alpha_decay <= 1.0)
        policy = ContinuousPolicy()
        super().__init__(
            policy = policy
        )
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.tau = tau

    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.alpha = self.alpha * self.alpha_decay
    
    def choose_action(
        self,
        state,
        # action_space = None, # deprecated
        phase = Phases.NONE
    ):
        action_space = self.env.action_space
        assert(type(action_space) is gym.spaces.Box)
        is_deterministic = (phase in [Phases.VALIDATION, Phases.TEST])
        if (phase in [Phases.TRAINING]):
            self.policy.train()
        elif (phase in [Phases.VALIDATION, Phases.TEST]):
            self.policy.eval()
        with torch.no_grad():
            state = torch.tensor(state)
            action = self.policy.sample(
                state,
                n_sample = 1,
                is_deterministic = is_deterministic
            ).detach().numpy()
        return action

    def choose_action_trajectory(
        self,
        state_trajectory,
        # action_space = None, # deprecated
        phase = Phases.NONE
    ):
        action_space = self.env.action_space
        assert(type(action_space) is gym.spaces.Box)
        is_deterministic = (phase in [Phases.VALIDATION, Phases.TEST])
        if (phase in [Phases.TRAINING]):
            self.policy.train()
        elif (phase in [Phases.VALIDATION, Phases.TEST]):
            self.policy.eval()
        with torch.no_grad():
            state_trajectory = torch.tensor(state_trajectory)
            action_trajectory = self.policy.sample(
                state_trajectory,
                n_sample = 1,
                is_deterministic = is_deterministic
            ).detach().numpy()
        return action_trajectory
    
    def update_policy(
        self,
        critic,
        trajectory,
        eps = 1e-8
    ):

        n_sample = 5

        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = trajectory

        action_trajectory = self.policy.sample(
            state_trajectory,
            n_sample = n_sample,
            requires_grad = True,
            is_deterministic = False
        )

        if (n_sample > 1):
            n_batch = state_trajectory.shape[0]
            state_trajectory = torch.tile(state_trajectory, dims=(n_sample, 1))
            action_trajectory = action_trajectory.reshape(n_sample * n_batch, -1)

        log_pi = self.policy.logP(state_trajectory, action_trajectory)
        q = critic.qvalue(state_trajectory, action_trajectory)

        loss = torch.sum(self.alpha * log_pi - q)
        if (torch.any(torch.isnan(loss))):
            assert(False)
        optim = self.policy.policy_optimizer
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    def update_target_policy(
        self,
        critic,
        trajectory
    ):
        for theta, target_theta in zip(self.policy.policy_network.parameters(), self.target_policy.policy_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

class SACCritic(Critic):

    def __init__(
        self,
        value = None,
        qvalue = None,
        gamma = 1.0,
        alpha = 1.0,
        tau = 0.5
    ):
        assert(0.0 < gamma <= 1.0)
        assert(alpha >= 0.0)
        assert(0.0 <= tau <= 1.0)
        value = Value()
        qvalue = ContinuousQValue()
        super().__init__(
            value = value,
            qvalue = qvalue
        )
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

    def update_qvalue(
        self,
        actor,
        trajectory,
        eps = 1e-8
    ):

        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = trajectory
        with torch.no_grad():
            reward_trajectory = reward_trajectory.reshape(-1, 1)

        q = self.qvalue(state_trajectory, action_trajectory)
        with torch.no_grad():            
            v = self.value(next_state_trajectory)

        loss = F.mse_loss(q, (reward_trajectory + self.gamma * v))
        if (torch.any(torch.isnan(loss))):
            assert(False)

        optim = self.qvalue.qvalue_optimizer
        optim.zero_grad()
        loss.backward()
        optim.step()

    def update_value(
        self,
        actor,
        trajectory,
        eps = 1e-8
    ):
        n_sample = 5

        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = trajectory
        with torch.no_grad():
            action_trajectory = actor.policy.sample(
                state_trajectory,
                n_sample = n_sample,
                requires_grad = False,
                is_deterministic = False
            )
        
        n_batch = state_trajectory.shape[0]
        v = self.value(state_trajectory)

        with torch.no_grad():
            if (n_sample > 1):
                state_trajectory = torch.tile(state_trajectory, dims=(n_sample, 1))
                action_trajectory = action_trajectory.reshape(n_sample * n_batch, -1)
            q = self.qvalue(state_trajectory, action_trajectory)
            log_pi = actor.policy.logP(state_trajectory, action_trajectory, eps=eps)
            q = q.reshape(n_sample, n_batch, 1)
            log_pi = log_pi.reshape(n_sample, n_batch, 1)
        
        loss = F.mse_loss(v, torch.sum(q - self.alpha * log_pi, dim=0))
        if (torch.any(torch.isnan(loss))):
            assert(False)
        optim = self.value.value_optimizer
        optim.zero_grad()
        loss.backward()
        optim.step()

    def update_target_qvalue(
        self,
        actor,
        trajectory
    ):
        for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

    def update_target_value(
        self,
        actor,
        trajectory
    ):
        for theta, target_theta in zip(self.value.value_network.parameters(), self.target_value.value_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data
    