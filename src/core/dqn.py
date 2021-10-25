# Deep Q Network (DQN)

import numpy as np
import torch
import torch.nn.functional as F
from src import policy

from value import Value, DiscreteQValue
from policy import Policy
from actor import Actor
from critic import Critic
from agent import Agent


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
        assert(model is not None)
        assert(memory is not None)
        actor = DQNActor(
            eps = eps,
            eps_decay = eps_decay
        )
        critic = DQNCritic(
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
        policy = Policy()
        super().__init__(
            policy = policy
        )
        self.copied_qvalue = None
        self.copied_target_qvalue = None
        assert(0.0 <= eps_decay <= 1.0)
        self.eps = eps
        self.eps_decay = eps_decay

    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.eps = self.eps * self.eps_decay
        print(self.eps)

    def choose_action(
        self,
        state,
        action_space
    ):
        if (self.copied_qvalue is None):
            action = action_space.sample()
        else:
            # epsilon-greedy
            q = self.copied_qvalue(torch.from_numpy(state)).detach().numpy()
            action = np.argmax(q)
            r = np.random.rand()
            if (r <= action_space.n * self.eps):
                action = action_space.sample()
        return action
    
    def update_policy(
        self,
        critic,
        trajectory = None
    ):
        self.copied_qvalue = critic.qvalue
        self.copied_target_qvalue = critic.target_qvalue

class DQNCritic(Critic):

    def __init__(
        self,
        value = None,
        qvalue = None,
        tau = 0.5,
        gamma = 1.0
    ):
        value = Value()
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
        q = torch.diag(self.qvalue(state_trajectory)[:, action_trajectory])
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
            target_theta.data = (1 - self.tau) * theta.data + self.tau * target_theta.data
