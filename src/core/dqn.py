# Deep Q Network (DQN)

import numpy as np
import torch
import torch.nn.functional as F

from ..policy import QValueBasedPolicy
from ..value import Value
from ..value import DiscreteQValue
from ..actor import DiscreteControlActorMixin
from ..critic import DiscreteControlCriticMixin
from ..critic import SoftUpdateCriticMixin
from ..agent import DiscreteControlAgentMixin


class DQNAgent(DiscreteControlAgentMixin):

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        gamma = 0.99,
        tau = 0.01,
        eps = 0.05,
        eps_decay = 1.0,
        use_default = False
    ):
        actor = DQNActor(
            eps = eps,
            eps_decay = eps_decay,
            interface = interface,
            use_default = use_default
        )
        critic = DQNCritic(
            gamma = gamma,
            tau = tau,
            interface = interface,
            use_default = use_default
        )
        DiscreteControlAgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = False
        )
        DQNAgent.setup(
            self,
            interface = None,
            configuration = None,
            actor = None,
            critic = None
        )

    def setup(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None
    ):
        self.actor.setup_with_critic(critic = self.critic)
        self.critic.setup_with_actor(actor = self.actor)
    
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.actor.setup_on_every_epoch(
            epoch = epoch,
            n_epoch = n_epoch
        )

class DQNActor(DiscreteControlActorMixin):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        eps = 0.0,
        eps_decay = 1.0,
        use_default = False
    ):
        policy = QValueBasedPolicy(
            interface = interface,
            configuration = configuration,
            use_default = use_default
        )
        DiscreteControlActorMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            policy = policy,
            use_default = False
        )
        self.eps = eps
        self.eps_decay = eps_decay

    def setup_with_critic(
        self,
        critic = None
    ):
        self.policy.setup_with_value(
            value = critic.value,
            qvalue = critic.qvalue
        )

    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        self.eps = self.eps * self.eps_decay

    def choose_action(
        self,
        state,
        information = None
    ):
        information["eps"] = self.eps
        action = self.policy.choose_action(
            state = state,
            information = information
        )
        return action
    
    def update_policy(
        self,
        critic,
        history = None
    ):
        pass

class DQNCritic(DiscreteControlCriticMixin, SoftUpdateCriticMixin):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        gamma = 0.99,
        tau = 0.5,
        use_default = False
    ):
        value = Value(
            interface = interface,
            configuration = configuration,
            use_default = use_default
        )
        qvalue = DiscreteQValue(
            interface = interface,
            configuration = configuration,
            use_default = use_default
        )
        DiscreteControlCriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            use_default = False
        )
        SoftUpdateCriticMixin.__init__(
            self,
            tau = tau
        )
        self.gamma = gamma

    def setup_with_actor(
        self,
        actor = None
    ):
        pass

    def update(
        self,
        actor,
        history,
        n_step = 1
    ):
        SoftUpdateCriticMixin.update(
            self,
            actor = actor,
            history = history,
            n_step = n_step
        )
    
    def update_qvalue(
        self,
        actor,
        history
    ):
        (state_trajectory, action_trajectory, reward_trajectory, next_state_trajectory) = history
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
        history = None
    ):
        SoftUpdateCriticMixin.update_target_qvalue(
            self,
            actor = actor,
            history = history
        )
