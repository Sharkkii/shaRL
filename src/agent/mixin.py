#### Agent (Mixin Class) ####

from ..common import Component
from ..common import SARS
from ..common import SGASG
from ..actor import Actor
from ..actor import DiscreteControlActor
from ..actor import ContinuousControlActor
from ..actor import GoalConditionedActor
from ..critic import Critic
from ..critic import DiscreteControlCritic
from ..critic import ContinuousControlCritic
from ..environment import GoalReachingTaskEnvironmentBase

from .base import AgentBase
from .base import DiscreteControlAgentBase
from .base import ContinuousControlAgentBase
from .base import GoalConditionedAgentBase


class AgentMixin(AgentBase, Component):

    def declare(self):
        self._interface = None
        self._configuration = None
        self._actor = None
        self._critic = None

    @property
    def interface(self): return self._interface
    @property
    def configuration(self): return self._configuration
    @property
    def actor(self): return self._actor
    @property
    def critic(self): return self._critic

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        allow_setup = True,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        if (default_actor is None):
            default_actor = Actor
        if (default_critic is None):
            default_critic = Critic

        AgentMixin.declare(self)
        Component.__init__(self)
        if (allow_setup):
            AgentMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                actor = actor,
                critic = critic,
                use_default = use_default,
                default_actor = default_actor,
                default_critic = default_critic
            )
    
    def setup(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        TACTOR = default_actor
        TCRITIC = default_critic

        if (use_default and (interface is None)):
            raise ValueError("`interface` must be 'AgentInterface' object if `use_default = True`.")
        if (configuration is None):
            # return
            configuration = {}
        if ((type(configuration) is not dict)):
            raise ValueError("`configuration` must be 'Dictionary' object or None.")

        if (use_default):
            if ((actor is not None) or (critic is not None)):
                raise ValueError("`actor` & `critic` must be None if `use_default = True`")
            if ((default_actor is None) or (default_critic is None)):
                raise ValueError("`default_actor` & `default_critic` must not be None if `use_default = True`")
            actor = TACTOR(
                interface = interface,
                configuration = configuration,
                use_default = True
            )
            critic = TCRITIC(
                interface = interface,
                use_default = True
            )
        
        else:
            if ((actor is None) or (critic is None)):
                return
        #         raise ValueError("`actor` & `critic` must not be None if `use_default = False`.")
        
        self._interface = interface
        self._configuration = configuration
        self._actor = actor
        self._critic = critic
        self._become_available()

    def epochwise_preprocess(
        self,
        epoch,
        n_epoch
    ):
        self.actor.epochwise_preprocess(
            epoch = epoch,
            n_epoch = n_epoch
        )
        self.critic.epochwise_preprocess(
            epoch = epoch,
            n_epoch = n_epoch
        )

    def epochwise_postprocess(
        self,
        epoch,
        n_epoch
    ):
        self.actor.epochwise_postprocess(
            epoch = epoch,
            n_epoch = n_epoch
        )
        self.critic.epochwise_postprocess(
            epoch = epoch,
            n_epoch = n_epoch
        )

    def stepwise_preprocess(
        self,
        step,
        n_step
    ):
        self.actor.stepwise_preprocess(
            step = step,
            n_step = n_step
        )
        self.critic.stepwise_preprocess(
            step = step,
            n_step = n_step
        )

    def stepwise_postprocess(
        self,
        step,
        n_step
    ):
        self.actor.stepwise_postprocess(
            step = step,
            n_estep= n_step
        )
        self.critic.stepwise_postprocess(
            step = step,
            n_estep= n_step
        )

    def choose_action(
        self,
        state,
        information = None
    ):
        return self.actor.choose_action(
            state = state,
            information = information
        )

    def interact_with_env(
        self,
        env,
        n_episode = 1,
        max_nstep = 1000,
        information = None,
        use_info = False,
        verbose = False,
    ):
        if (information is None):
            information = {}
        if (type(information) is not dict):
            raise ValueError("`information` must be a 'dictionary' object.")
        information["action_space"] = env.action_space

        history = []
        info_history = []

        for _ in range(n_episode):

            t = 0
            state = env.reset()
            done = False
            for _ in range(max_nstep):
                if (done): break
                action = self.actor.choose_action(
                    state = state,
                    information = information
                )
                next_state, reward, done, info = env.step(action)
                data = SARS(state, action, reward, next_state)
                history.append(data)
                state = next_state
                if (use_info):
                    info_history.append(info)
                t = t + 1
        
        if (use_info):
            return history, info_history
        else:
            return history

    def update_actor(
        self,
        history,
        n_step = 1
    ):
        return self.actor.update(
            critic = self.critic,
            history = history,
            n_step = n_step
        )

    def update_critic(
        self,
        history,
        n_step = 1
    ):
        return self.critic.update(
            actor = self.actor,
            history = history,
            n_step = n_step
        )

    def train(
        self
    ):
        self.actor.train()
        self.critic.train()

    def eval(
        self
    ):
        self.actor.eval()
        self.critic.eval()


class DiscreteControlAgentMixin(AgentMixin, DiscreteControlAgentBase):
    
    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        allow_setup = True,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        if (default_actor is None):
            default_actor = DiscreteControlActor
        if (default_critic is None):
            default_critic = DiscreteControlCritic

        AgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = use_default,
            default_actor = default_actor,
            default_critic = default_critic
        )
        if (allow_setup):
            DiscreteControlAgentMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                actor = actor,
                critic = critic,
                use_default = use_default,
                default_actor = default_actor,
                default_critic = default_critic
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        pass


class ContinuousControlAgentMixin(AgentMixin, ContinuousControlAgentBase):
    
    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        allow_setup = True,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        if (default_actor is None):
            default_actor = ContinuousControlActor
        if (default_critic is None):
            default_critic = ContinuousControlCritic

        AgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            use_default = use_default,
            default_actor = default_actor,
            default_critic = default_critic
        )
        if (allow_setup):
            ContinuousControlAgentMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                actor = actor,
                critic = critic,
                use_default = use_default,
                default_actor = default_actor,
                default_critic = default_critic
            )

    def setup(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        pass


class GoalConditionedAgentMixin(AgentMixin, GoalConditionedAgentBase):

    def __init__(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        allow_setup = True,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        if (default_actor is None):
            default_actor = GoalConditionedActor
        if (default_critic is None):
            default_critic = Critic

        AgentMixin.__init__(
            self,
            interface = interface,
            configuration = configuration,
            actor = actor,
            critic = critic,
            allow_setup = allow_setup,
            use_default = use_default,
            default_actor = default_actor,
            default_critic = default_critic
        )
        if (allow_setup):
            GoalConditionedAgentMixin.setup(
                self,
                interface = interface,
                configuration = configuration,
                actor = actor,
                critic = critic,
                use_default = use_default,
                default_actor = default_actor,
                default_critic = default_critic
            )
    
    def setup(
        self,
        interface = None,
        configuration = None,
        actor = None,
        critic = None,
        use_default = False,
        default_actor = None,
        default_critic = None
    ):
        pass

    def choose_action(
        self,
        state,
        goal,
        information = None
    ):
        return self.actor.choose_action(
            state = state,
            goal = goal,
            information = information
        )

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
        if (not isinstance(env, GoalReachingTaskEnvironmentBase)):
            raise ValueError("`env` must be 'GoalReachingTaskEnvironment' object.")

        if (information is None):
            information = {}
        if (type(information) is not dict):
            raise ValueError("`information` must be a 'dictionary' object.")
        information["action_space"] = env.action_space

        # assume `use_goal = True & use_reward = False`
        if (not(use_goal)):
            raise NotImplementedError("`use_goal` must be True.")
        if (use_reward):
            raise NotImplementedError("`use_reward` must be False.")
        
        history = []
        info_history = []

        for _ in range(n_episode):

            t = 0
            state, goal = env.reset(use_goal = True)
            done = False
            for _ in range(max_nstep):
                if (done): break
                action = self.actor.choose_action(
                    state = state,
                    goal = goal,
                    information = information
                )
                # assume `use_goal = True & use_reward = False`
                next_state, goal, done, info = env.step(action)
                next_goal = goal
                data = SGASG(state, goal, action, next_state, next_goal)
                history.append(data)
                state = next_state
                if (use_info):
                    info_history.append(info)
                t = t + 1
        
        if (use_info):
            return history, info_history
        else:
            return history