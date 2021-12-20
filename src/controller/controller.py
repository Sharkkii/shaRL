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
        n_epoch = 1,
        n_sample = 1,
        n_sample_start = 1,
        n_eval = None, # deprecated
        n_train_eval = 1,
        n_test_eval = 0,
        n_eval_interval = 10,
        env_step = 1,
        gradient_step = -1 # supported only when `gradient_step` < 0
    ):
        # is_model_based = self.props_agent["is_model_based"]
        # is_model_free = self.props_agent["is_model_free"]
        # is_value_based = self.props_agent["is_value_based"]
        # is_policy_based = self.props_agent["is_policy_based"]
        # is_on_policy = self.props_agent["is_on_policy"]
        # is_off_policy = self.props_agent["is_off_policy"]
        # is_online = self.props_agent["is_online"]

        # TODO: -> agent.setup (see props.py)
        # is_deterministic_policy = self.props_agent["is_deterministic_policy"]

        self.agent.train()

        if (gradient_step > 0):
            warnings.warn("`gradient_step` controlls how many batches we use to update parameters. It's better to use the default value -1 (use all batches).")

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
        
        is_deterministic_policy = False

        assert(n_sample <= n_sample_start <= self.agent.memory.capacity)

        # FIXME: define dataset & dataloader
        from ..memory import RLDataset, RLDataLoader
        from ..memory import TensorConverter, RewardStabilizer

        n_batch = 100
        transform = TensorConverter()
        dataset = RLDataset(
            min_size = 1000,
            max_size = 100000,
            transform = transform
        )
        dataloader = RLDataLoader(
            dataset = dataset,
            batch_size = n_batch,
            shuffle = True
        )

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

                # interact w/ environment
                if (is_model_free):
                    trajs_env = self.agent.interact_with(
                        self.env,
                        n_times = 1,
                        phase = Phases.TRAINING
                    )

                # interact w/ model    
                if (is_model_based):
                    trajs_model = self.agent.interact_with(
                        self.agent.model,
                        n_times = 1,
                        phase = Phases.TRAINING
                    )

            trajs = trajs_env + trajs_model
            dataloader.save(trajs)
            if (not dataloader.is_available):
                continue

            # if (is_on_policy):
            #     pass # convert trajs by Memory.zip

            # if (is_off_policy):
            #     self.agent.save_history(trajs)
            #     # guard
            #     if (self.agent.memory.count >= n_sample_start):
            #         trajs = self.agent.replay_history(
            #             n_sample = n_sample
            #         )
            #         # trajs = self.agent.load_history()
            #     else:
            #         continue

            # optimize policy & value
            if (gradient_step <= 0):
                
                # use all data
                # NOTE: history: {(s,a,r,s)}
                for history in dataloader:

                    # guard
                    if (len(history[0]) < n_batch):
                        continue

                    # update value function (critic)
                    self.agent.update_critic(history, n_times=1)

                    # update policy (actor)
                    self.agent.update_actor(history, n_times=1)

                    # learn dynamics
                    # if (is_model_based):
                    #     self.agent.update_model(history, n_times=1)

            else:
                assert(False)

                # use the number (designated by `gradient_step`) of batches
                # NOT supported now
                for _ in range(gradient_step):

                    # update value function (critic)
                    self.agent.update_critic(trajs, n_times=1)

                    # update policy (actor)
                    self.agent.update_actor(trajs, n_times=1)

                    # learn dynamics
                    if (is_model_based):
                        self.agent.update_model(trajs, n_times=1)

            # TODO: learn something if needed
            # self.agent.update_every_epoch(
            #     epoch=epoch,
            #     n_epoch=n_epoch
            # )

            # evaluate
            if ((epoch+1) % n_eval_interval == 0):
                if (n_train_eval > 0 or n_test_eval > 0):
                    train_score, test_score = self.evaluate(
                        n_train_eval = n_train_eval,
                        n_test_eval = n_test_eval
                    )
                print((epoch+1), train_score, test_score)
            else:
                print("\r%d" % (epoch+1), end="")
            

    def evaluate(
        self,
        n_train_eval = 1,
        n_test_eval = 0
    ):
        self.agent.eval()

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
