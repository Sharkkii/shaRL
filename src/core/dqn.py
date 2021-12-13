# Deep Q Network (DQN)

import numpy as np
import torch
import torch.nn.functional as F

from ..value import Value, DiscreteQValue
from ..policy import QBasedPolicy
from ..actor import Actor
from ..critic import Critic
from ..agent import Agent
from ..controller import Phases


class DQN(Agent):

    def __init__(
        self,
        actor = None,
        critic = None,
        model = None,
        memory = None,
        gamma = 1.0,
        tau = 0.5,
        eps = 0.0,
        eps_decay = 1.0
    ):
        actor = DQNActor(
            eps = eps,
            eps_decay = eps_decay
        )
        critic = DQNCritic(
            gamma = gamma,
            tau = tau
        )
        super().__init__(
            actor = actor,
            critic = critic,
            model = model,
            memory = memory,
            gamma = gamma
        )
    
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.actor.setup_on_every_epoch(
            epoch = epoch,
            n_epoch = n_epoch
        )

class DQNActor(Actor):

    def __init__(
        self,
        policy = None,
        eps = 0.0,
        eps_decay = 1.0
    ):
        assert(0.0 <= eps_decay <= 1.0)
        policy = QBasedPolicy(
            copied_qvalue = None
        )
        super().__init__(
            policy = policy
        )
        self.eps = eps
        self.eps_decay = eps_decay

    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.eps = self.eps * self.eps_decay
        # print(self.eps)

    def choose_action(
        self,
        state,
        phase = Phases.NONE
    ):
        action = self.policy.sample(
            state = state,
            action_space = self.env.action_space,
            phase = phase,
            eps = self.eps
        )
        return action
    
    def update_policy(
        self,
        critic,
        trajectory = None
    ):
        self.policy.copied_qvalue = critic.qvalue
        self.policy.copied_target_qvalue = critic.target_qvalue

class DQNCritic(Critic):

    def __init__(
        self,
        value = None,
        qvalue = None,
        gamma = 1.0,
        tau = 0.5
    ):
        assert(0.0 < gamma <= 1.0)
        assert(0.0 <= tau <= 1.0)
        value = None
        qvalue = DiscreteQValue()
        super().__init__(
            value = value,
            qvalue = qvalue
        )
        self.tau = tau
        self.gamma = gamma
    
    def update_qvalue(
        self,
        actor,
        trajectory
    ):
        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = trajectory
        action_trajectory = action_trajectory.long()

        # q = torch.diag(self.qvalue(state_trajectory)[:, action_trajectory])
        batch_size = len(state_trajectory)
        q = torch.cat([self.qvalue(state_trajectory)[[n], [action_trajectory[n]]] for n in range(batch_size)], axis=0)
        with torch.no_grad():
            target_q, _ = torch.max(self.target_qvalue(next_state_trajectory), dim=1)
            target_q = reward_trajectory + self.gamma * target_q

        optim = self.qvalue.qvalue_optimizer
        optim.zero_grad()
        loss = F.mse_loss(q, target_q)
        loss.backward()
        optim.step()

    def update_target_qvalue(
        self,
        actor,
        trajectory
    ):
        for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data
