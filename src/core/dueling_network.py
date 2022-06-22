# Deep Q Network (DuelingNetwork)

import torch
import torch.nn.functional as F

from ..common import ValueReference
from ..common import AdvantageReference
from ..policy import DuelingNetworkQValueBasedEpsilonGreedyPolicy
from ..value import Value
from ..value import DiscreteDuelingNetworkQValue
from ..value import DiscreteAdvantage
from ..actor import DiscreteControlActorMixin
from ..critic import DiscreteControlCriticMixin
from ..critic import SoftUpdateCriticMixin
from ..agent import DiscreteControlAgentMixin


class DuelingNetworkAgent(DiscreteControlAgentMixin):

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
        actor = DuelingNetworkActor(
            interface = interface,
            configuration = { "epsilon": eps },
            use_default = use_default
        )
        critic = DuelingNetworkCritic(
            gamma = gamma,
            tau = tau,
            interface = interface,
            configuration = None,
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
        DuelingNetworkAgent.setup(
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
  

class DuelingNetworkActor(DiscreteControlActorMixin):

    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = False
    ):
        policy = DuelingNetworkQValueBasedEpsilonGreedyPolicy(
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

    def setup_with_critic(
        self,
        critic = None
    ):
        self.policy.setup_with_value(
            qvalue = critic.qvalue,
        )

    def choose_action(
        self,
        state,
        information = None
    ):
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

class DuelingNetworkCritic(DiscreteControlCriticMixin, SoftUpdateCriticMixin):

    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        gamma = 0.99,
        tau = 0.5,
        use_default = False
    ):
        value = Value(
            interface = interface,
            configuration = configuration,
            use_default = use_default
        )
        advantage = DiscreteAdvantage(
            interface = interface,
            configuration = configuration,
            use_default = use_default
        )
        qvalue = DiscreteDuelingNetworkQValue(
            interface = interface,
            configuration = configuration,
            value_reference = value,
            advantage_reference = advantage,
            use_default = use_default
        )
        DiscreteControlCriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            advantage = advantage,
            use_default = False
        )
        SoftUpdateCriticMixin.__init__(
            self,
            tau = tau
        )
        DuelingNetworkCritic.setup(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            advantage = advantage,
            use_default = False
        )
        self.gamma = gamma

    def setup(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        advantage = None,
        use_default = False
    ):
        # FIXME:
        self._target_qvalue._value_reference = ValueReference(
            target = self._target_value
        )
        self._target_qvalue._advantage_reference = AdvantageReference(
            target = self._target_advantage
        )

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

    def update_value(
        self,
        actor,
        history
    ):
        pass
    
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

        value = self.value
        advantage = self.advantage
        optim_v = value.value_optimizer
        optim_adv = advantage.advantage_optimizer

        optim_v.zero_grad()
        optim_adv.zero_grad()
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()

        optim_v.clip_grad_value(value = 1.0)
        optim_adv.clip_grad_value(value = 1.0)
        optim_v.step()
        optim_adv.step()


    def update_advantage(
        self,
        actor,
        history
    ):
        pass

    def update_target_value(
        self,
        actor,
        history = None
    ):
        SoftUpdateCriticMixin.update_target_value(
            self,
            actor = actor,
            history = history
        )

    def update_target_qvalue(
        self,
        actor,
        history = None
    ):
        pass

    def update_target_advantage(
        self,
        actor,
        history = None
    ):
        SoftUpdateCriticMixin.update_target_advantage(
            self,
            actor = actor,
            history = history
        )
