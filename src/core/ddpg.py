#### Deep Deterministic Policy Gradient (DDPG) ####

import numpy as np
import torch
import torch.nn.functional as F
import gym

from value import Value, ContinuousQValue
from policy import Policy
from actor import Actor
from critic import Critic
from agent import Agent


class DeepDeterministicPolicyGradient(Agent):

    def __init__(
        self,
        actor = None,
        critic = None,
        model = None,
        memory = None,
        gamma = 1.0,
        tau = 0.5
    ):
        assert(model is not None)
        assert(memory is not None)
        actor = DDPGActor(
            tau = tau
        )
        critic = DDPGCritic(
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
    
class DDPGActor(Actor):

    def __init__(
        self,
        policy = None,
        tau = 0.5
    ):
        policy = Policy()
        super().__init__(
            policy = policy
        )
        self.tau = tau
    
    def choose_action(
        self,
        state,
        action_space,
        is_deterministic = False
    ):
        assert(type(action_space) is gym.spaces.Box)
        action = self.policy(torch.from_numpy(state)).detach().numpy()
        if (not is_deterministic):
            action += np.random.randn()
        return action
    
    def update_policy(
        self,
        critic,
        trajectory
    ):
        (state_trajectory, _, _, _) = trajectory

        action_trajectory = self.policy(state_trajectory)
        q = critic.qvalue(state_trajectory, action_trajectory)
        sum_q = torch.sum(q)

        policy_optimizer = self.policy.policy_optimizer
        policy_optimizer.zero_grad()
        sum_q.backward()
        policy_optimizer.step()

    def update_target_policy(
        self,
        critic,
        trajectory
    ):
        for theta, target_theta in zip(self.policy.policy_network.parameters(), self.target_policy.policy_network.parameters()):
            target_theta.data = (1 - self.tau) * target_theta.data + self.tau * theta.data

class DDPGCritic(Critic):

    def __init__(
        self,
        value = None,
        qvalue = None,
        gamma = 1.0,
        tau = 0.5
    ):
        assert(0.0 < gamma <= 1.0)
        value = Value()
        qvalue = ContinuousQValue()
        super().__init__(
            value = value,
            qvalue = qvalue
        )
        self.gamma = gamma
        self.tau = tau
    
    def update_qvalue(
        self,
        actor,
        trajectory
    ):
        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = trajectory
        reward_trajectory = reward_trajectory.reshape(-1,1)

        q = self.qvalue(state_trajectory, action_trajectory)
        with torch.no_grad():
            next_action_trajectory = actor.policy(state_trajectory)
            target_q = reward_trajectory + self.gamma * self.target_qvalue(next_state_trajectory, next_action_trajectory)

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