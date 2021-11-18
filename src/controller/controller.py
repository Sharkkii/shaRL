#### Controller ####

import warnings
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import torch

# from props import props_env, props_agent


class BaseController(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(
        self,
        environment,
        agent,
        config = None
    ):
        raise NotImplementedError
    
    @abstractmethod
    def setup(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

class Controller(BaseController):

    def __init__(
        self,
        environment,
        agent,
        config={
            "env_name": "Default",
            "agent_name": "Default"
        }
    ):

        self.env = environment
        self.agent = agent
        self.config = config

        # env_name = config["env_name"]
        # agent_name = config["agent_name"]
        # self.props_env = props_env[env_name]
        # self.props_agent = props_agent[agent_name]
    
    def setup(
        self
    ):
        self.env.setup()
        self.agent.setup()

    def reset(
        self
    ):
        self.env.reset()
        self.agent.reset()
    
    def fit(
        self,
        n_epoch=1,
        n_sample=1,
        n_sample_start=1,
        n_eval = None, # deprecated
        n_train_eval = 1,
        n_test_eval = 0,
        n_eval_interval = 1,
        env_step=1,
        gradient_step=1,
    ):
        # is_model_based = self.props_agent["is_model_based"]
        # is_model_free = self.props_agent["is_model_free"]
        # is_value_based = self.props_agent["is_value_based"]
        # is_policy_based = self.props_agent["is_policy_based"]
        # is_on_policy = self.props_agent["is_on_policy"]
        # is_off_policy = self.props_agent["is_off_policy"]
        # is_online = self.props_agent["is_online"]

        if (n_eval is not None):
            warnings.warn("`n_eval` is deprecated. Use `n_train_eval` & `n_test_eval` instead.")
            n_train_eval = n_eval
            n_test_eval = 0

        is_model_based = False
        is_model_free = True
        is_value_based = True
        is_policy_based = False
        is_on_policy = False
        is_off_policy = True
        is_online = False
        assert(is_on_policy == (not is_off_policy))

        # online
        if (is_online):
            env_step = 1
            is_model_based = False
            is_on_policy = True
            is_off_policy = False

        # TODO: -> agent.setup (see props.py)
        # is_discrete_state_space = self.props_env["is_discrete_state_space"]
        # is_discrete_action_space = self.props_env["is_discrete_action_space"]
        # is_deterministic_policy = self.props_agent["is_deterministic_policy"]
        is_discrete_state_space = False
        is_discrete_action_space = True
        is_deterministic_policy = False

        assert(n_sample <= n_sample_start <= self.agent.memory.capacity)

        for epoch in range(n_epoch):

            self.agent.setup_on_every_epoch(
                epoch = epoch,
                n_epoch = n_epoch
            )

            # train

            # interact & prepare training data
            trajs_env = []
            trajs_model = []

            for step in range(env_step):

                self.agent.setup_on_every_step(
                    step = step,
                    n_step = env_step
                )

                # print("environment_step")

                # interact w/ environment
                if (is_model_free):
                    trajs_env = self.agent.interact_with(
                        self.env,
                        # is_discrete_action_space=is_discrete_action_space,
                        # is_deterministic_policy=is_deterministic_policy,
                        n_times = 1
                    )

                # interact w/ model    
                if (is_model_based):
                    trajs_model = self.agent.interact_with(
                        self.agent.model,
                        # is_discrete_action_space=is_discrete_action_space,
                        # is_deterministic_policy=is_deterministic_policy,
                        n_times = 1
                    )

            trajs = trajs_env + trajs_model
            if (is_on_policy):
                pass # convert trajs by Memory.zip

            if (is_off_policy):
                self.agent.save_history(trajs)
                # guard
                if (self.agent.memory.count >= n_sample_start):
                    trajs = self.agent.replay_history(
                        n_sample = n_sample
                    )
                    # trajs = self.agent.load_history()
                else:
                    continue

            # optimize policy
            # FIXME: J_v, J_q, J_pi will be overwritten
            for _ in range(gradient_step):

                # print("gradient_step")

                # update value function (critic)
                self.agent.update_critic(trajs, n_times=1)
                # J_v, J_q = self.agent.update_critic(trajs, n_times=1)

                # update policy (actor)
                self.agent.update_actor(trajs, n_times=1)
                # J_pi = self.agent.update_actor(trajs, n_times=1)

                # learn dynamics
                if (is_model_based):
                    self.agent.update_model(trajs, n_times=1)
                    # J_m = self.agent.update_model(n_times=1)
                else:
                    # J_m = None
                    pass

            # TODO: learn something if needed
            # self.agent.update_every_epoch(
            #     epoch=epoch,
            #     n_epoch=n_epoch
            # )

            # evaluate
            if (n_train_eval > 0 or n_test_eval > 0):
                train_score, test_score = self.evaluate(
                    n_train_eval = n_train_eval,
                    n_test_eval = n_test_eval
                )
                print(train_score, test_score)
            
            # J_v = J_q = J_pi = J_m = 0
            # print("%d" % epoch, end="\r", flush=True)
            # if ((epoch+1) % n_eval_interval == 0):
            #     print(epoch+1, end=" | ")
            #     result = self.evaluate(n_eval)
            #     if (J_v is not None):
            #         print("J_v: %2.4f" % J_v, end=" | ")
            #     else:
            #         print(J_v, end=" | ")
            #     if (J_q is not None):
            #         print("J_q: %2.4f" % J_q, end=" | ")
            #     else:
            #         print(J_q, end=" | ")
            #     if (J_pi is not None):
            #         print("J_pi: %2.4f" % J_pi, end=" | ")
            #     else:
            #         print(J_pi,  end=" | ")
            #     print("duration: %d" % result[0], end=" ")
            #     print("total reward: %f" % result[1], end=" ")
            #     print()

    def evaluate(
        self,
        n_train_eval = 1,
        n_test_eval = 0
    ):
        # initialize score dictionary
        train_score = self.env.score([])
        for key, _ in train_score.items():
            train_score[key] = []
        test_score = self.env.score([])
        for key, _ in test_score.items():
            test_score[key] = []
        
        # evaluate (train)
        for _ in range(n_train_eval):
            traj = self.agent.interact_with(
                self.env,
                n_times = 1,
                phase = Phases.TRAINING
            )
            score = self.env.score(traj)
            for key, value in score.items():
                train_score[key].append(value)
        
        # evlauate (test)
        for _ in range(n_test_eval):
            traj = self.agent.interact_with(
                self.env,
                n_times = 1,
                phase = Phases.TEST
            )
            score = self.env.score(traj)
            for key, value in score.items():
                test_score[key].append(value)

        return (train_score, test_score)

class Phases(Enum):
    NONE = 0
    TRAINING = 1
    VALIDATION = 2
    TEST = 3