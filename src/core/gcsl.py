#### Goal-conditioned Supervised-learning (GCSL)

from cv2 import log
import torch
import torch.nn.functional as F


from ..const import PhaseType
from ..common import SGASG
from ..policy import GoalConditionedDiscretePolicy
from ..actor import DiscreteControlActorMixin
from ..actor import GoalConditionedActorMixin
from ..critic import DiscreteControlCriticMixin
from ..agent import DiscreteControlAgentMixin
from ..agent import GoalConditionedAgentMixin


class GCSLAgent(GoalConditionedAgentMixin, DiscreteControlAgentMixin):

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False
    ):
        actor = GCSLActor(
            interface = interface,
            configuration = None,
            use_default = use_default
        )
        critic = GCSLCritic(
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
        GCSLAgent.setup(
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
        self.meta_policy = "greedy"

    def epochwise_preprocess(
        self,
        epoch,
        n_epoch
    ):
        if (epoch < n_epoch//5):
            self.meta_policy = "random"
        else:
            self.meta_policy = "mixed"

    def interact_with_env(
        self,
        env,
        n_episode = 1,
        max_nstep = 1000,
        information = None,
        use_info = False,
        use_goal = True, # will be implemented
        use_reward = False, # will be implemented
        verbose = False
    ):
        if (information is None):
            information = { "phase": PhaseType.TRAINING }
        information["meta_policy"] = self.meta_policy
        information["action_space"] = env.action_space
        
        if (use_info):
            history, info_history = GoalConditionedAgentMixin.interact_with_env(
                self,
                env = env,
                n_episode = n_episode,
                max_nstep = max_nstep,
                information = information,
                use_info = use_info,
                use_goal = use_goal,
                use_reward = use_reward,
                verbose = False
            )
            if (information["phase"] is PhaseType.TRAINING):
                history = self.relabel_in_hindsight(history, use_original = True)
            return history, info_history

        else:
            history = GoalConditionedAgentMixin.interact_with_env(
                self,
                env = env,
                n_episode = n_episode,
                max_nstep = max_nstep,
                information = information,
                use_info = use_info,
                use_goal = use_goal,
                use_reward = use_reward,
                verbose = False
            )
            if (information["phase"] is PhaseType.TRAINING):
                history = self.relabel_in_hindsight(history, use_original = True)
            return history
        
    def relabel_in_hindsight(
        self,
        history,
        use_original = True
    ):
        relabeled_dataset = []
        if (use_original):
            relabeled_dataset.extend(history)

        T = len(history)
        K = 20
        for te in range(max(0, T - K), T):
            for ts in range(T):

                sags_ts = history[ts]
                sags_te = history[te]
                relabeled_data = SGASG(
                    state = sags_ts.state,
                    goal = sags_te.state,
                    action = sags_ts.action,
                    next_state = sags_ts.next_state,
                    next_goal = sags_te.state
                )
                relabeled_dataset.append(relabeled_data)

        return relabeled_dataset


class GCSLActor(GoalConditionedActorMixin, DiscreteControlActorMixin):
    
    def __init__(
        self,
        interface = None,
        configuration = None,
        policy = None,
        use_default = False
    ):
        policy = GoalConditionedDiscretePolicy(
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

    def __call__(
        self,
        state,
        goal,
        information = None
    ):
        return GoalConditionedActorMixin.choose_action(
            self,
            state = state,
            goal = goal,
            information = information
        )

    def choose_action(
        self,
        state,
        goal,
        information = None
    ):
        action = GoalConditionedActorMixin.choose_action(
            self,
            state = state,
            goal = goal,
            information = information
        )
        return action

    def update_policy(
        self,
        critic,
        history
    ):
        (state_history, goal_history, action_history, _, _) = history
        action_history = action_history.long()

        total_log_likelihood = 0.0

        for idx, (state, goal, action) in enumerate(zip(state_history, goal_history, action_history)):

            P = self.policy.P(
                state = state,
                goal = goal
            )
            logP = self.policy.logP(
                state = state,
                goal = goal
            )
            log_likelihood = logP[action]
            total_log_likelihood = total_log_likelihood + log_likelihood

        T = len(state_history)
        total_log_likelihood = total_log_likelihood / T

        nloss = total_log_likelihood
        loss = -1 * nloss
        # print("%3.3f | %d %d %d" % (loss.item(), sum([action == 0 for action in action_history]).item(), sum([action == 1 for action in action_history]).item(), sum([action == 2 for action in action_history]).item()))

        optim = self.policy.policy_optimizer
        optim.zero_grad()
        loss.backward()
        optim.clip_grad_value(value = 1.0)
        optim.step()


class GCSLCritic(DiscreteControlCriticMixin):
    
    def __init__(
        self,
        interface = None,
        configuration = None,
        value = None,
        qvalue = None,
        use_default = False
    ):
        DiscreteControlCriticMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            value = value,
            qvalue = qvalue,
            use_default = use_default
        )
