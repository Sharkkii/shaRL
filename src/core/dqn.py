# Deep Q Network (DQN)

import numpy as np
import torch
import torch.nn.functional as F

from ..const import PhaseType
from ..policy import QBasedPolicy
from ..value import PseudoValue
from ..value import DiscreteQValue
from ..actor import Actor
from ..critic import Critic
from ..agent import Agent


class DQN(Agent):

    def __init__(
        self,
        actor = None,
        critic = None,
        model = None,
        memory = None,
        gamma = 0.99,
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
        policy = QBasedPolicy()
        super().__init__(
            policy = policy
        )
        self.eps = eps
        self.eps_decay = eps_decay

    def setup_with_critic(
        self,
        critic
    ):
        self.policy.setup_reference_qvalue(critic.qvalue)

    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.eps = self.eps * self.eps_decay

    def choose_action(
        self,
        state,
        phase = PhaseType.NONE
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
        self.policy.setup_reference_qvalue(critic.qvalue)

class DQNCritic(Critic):

    def __init__(
        self,
        value = None,
        qvalue = None,
        gamma = 0.99,
        tau = 0.5
    ):
        assert(0.0 < gamma <= 1.0)
        assert(0.0 <= tau <= 1.0)
        value = PseudoValue()
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
        batch_size = len(state_trajectory)

        q = self.qvalue(state_trajectory)
        y_pred = torch.cat([q[[n], [action_trajectory[n]]] for n in range(batch_size)], axis=0)

        with torch.no_grad():
            target_q, _ = torch.max(self.target_qvalue(next_state_trajectory), dim=1)
            y_true = reward_trajectory + self.gamma * target_q

        optim = self.qvalue.qvalue_optimizer
        optim.zero_grad()
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optim.clip_grad_value(value = 1.0)
        optim.step()

    def update_target_qvalue(
        self,
        actor,
        trajectory
    ):
        for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data
