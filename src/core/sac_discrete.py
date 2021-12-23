#### Soft Actor Critic Discrete (SAC-Discrete) ####

import numpy as np
import torch
import torch.nn.functional as F
import gym

from ..const import PhaseType
from ..value import Value, DiscreteQValue
from ..policy import Policy
from ..actor import Actor
from ..critic import Critic
from ..agent import Agent


class SoftActorCriticDiscrete(Agent):

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
        actor = SACDiscreteActor(
            alpha = alpha,
            alpha_decay = alpha_decay
        )
        critic = SACDiscreteCritic(
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

class SACDiscreteActor(Actor):

    def __init__(
        self,
        policy = None,
        alpha = 1.0,
        alpha_decay = 1.0
    ):
        assert(alpha >= 0.0)
        assert(0.0 < alpha_decay <= 1.0)
        policy = Policy()
        super().__init__(
            policy = policy
        )
        self.alpha = alpha
        self.alpha_decay = alpha_decay

    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.alpha = self.alpha * self.alpha_decay
    
    def choose_action(
        self,
        state,
        phase = PhaseType.NONE
    ):
        action_space = self.env.action_space
        assert(type(action_space) is gym.spaces.Discrete)

        with torch.no_grad():
            p = self.policy.P(torch.from_numpy(state)).detach().numpy()

        is_deterministic = (phase in [PhaseType.VALIDATION, PhaseType.TEST])
        if is_deterministic:
            action = np.argmax(p)
        else:
            action = np.random.choice(action_space.n, p=p)
        return action
    
    def update_policy(
        self,
        critic,
        trajectory,
        eps = 1e-8
    ):

        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = trajectory
        action_trajectory = action_trajectory.long()

        with torch.no_grad():
            q = critic.qvalue(state_trajectory)
        pi = torch.clamp(self.policy.P(state_trajectory), eps, 1 - eps)
        kl_divergence = torch.sum(pi * (self.alpha * torch.log(pi) - q), dim=1)

        optim = self.policy.policy_optimizer
        optim.zero_grad()
        loss = torch.mean(kl_divergence)
        loss.backward()
        optim.clip_grad_norm(value = 1.0)
        optim.step()

class SACDiscreteCritic(Critic):

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
        qvalue = DiscreteQValue()
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
        action_trajectory = action_trajectory.long()
        batch_size = len(state_trajectory)

        q = self.qvalue(state_trajectory)
        y_pred = torch.cat([q[[n], [action_trajectory[n]]] for n in range(batch_size)], axis=0)

        with torch.no_grad():
            pi = torch.clamp(actor.policy.P(next_state_trajectory), eps, 1 - eps)
            target_q = self.target_qvalue(next_state_trajectory)
            v = torch.sum(pi * (target_q - self.alpha * torch.log(pi)), dim=1)
            y_true = reward_trajectory + self.gamma * v

        optim = self.qvalue.qvalue_optimizer
        optim.zero_grad()
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optim.clip_grad_norm(value = 1.0)
        optim.step()

    def update_target_qvalue(
        self,
        actor,
        trajectory
    ):
        for theta, target_theta in zip(self.qvalue.qvalue_network.parameters(), self.target_qvalue.qvalue_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

    