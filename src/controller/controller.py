#### Controller ####

import warnings
from abc import ABCMeta, abstractmethod

from ..const import PhaseType
from ..const import EnvironmentModelType, AgentStrategyType, AgentBehaviorType, AgentLearningType
from ..memory import RLDataset, RLDataLoader
from ..memory import TensorConverter


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
        config = {}
    ):

        self.env = environment
        self.agent = agent
        self.config = config
    
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
        gradient_step = -1, # supported only when `gradient_step` < 0
        dataset_size = 100000,
        batch_size = 100,
        shuffle = True,
        return_score = True,
    ):

        if (gradient_step > 0):
            warnings.warn("`gradient_step` controlls how many batches we use to update parameters. It's better to use the default value -1 (use all batches).")

        if (n_eval is not None):
            warnings.warn("`n_eval` is deprecated. Use `n_train_eval` & `n_test_eval` instead.")
            n_train_eval = n_eval
            n_test_eval = 0

        model_type = EnvironmentModelType.MODEL_FREE
        strategy_type = AgentStrategyType.VALUE_BASED
        behavior_type = AgentBehaviorType.OFF_POLICY
        learning_type = AgentLearningType.OFFLINE

        # assert(n_sample <= n_sample_start <= self.agent.memory.capacity)

        # FIXME: define dataset & dataloader
        transform = TensorConverter()
        dataset = RLDataset(
            min_size = batch_size,
            max_size = dataset_size,
            transform = transform
        )
        dataloader = RLDataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )

        metrics = self.env.score([]).keys()
        train_score_history = { metric: [] for metric in metrics }
        test_score_history = { metric: [] for metric in metrics }

        self.agent.train()

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
                if (model_type == EnvironmentModelType.MODEL_FREE):
                    trajs_env = self.agent.interact_with(
                        self.env,
                        n_times = 1,
                        phase = PhaseType.TRAINING
                    )

                # interact w/ model    
                if (model_type == EnvironmentModelType.MODEL_BASED):
                    trajs_model = self.agent.interact_with(
                        self.agent.model,
                        n_times = 1,
                        phase = PhaseType.TRAINING
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
                    if (len(history[0]) < batch_size):
                        continue

                    # update value function (critic)
                    self.agent.update_critic(history, n_times=1)

                    # update policy (actor)
                    self.agent.update_actor(history, n_times=1)

                    # learn dynamics
                    # if (model_type == EnvironmentModelType.MODEL_BASED):
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
                    if (model_type == EnvironmentModelType.MODEL_BASED):
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
                print("\r%d" % (epoch+1))
                self.report(train_score, test_score)
                if (return_score):
                    for key, value in train_score.items():
                        train_score_history[key].append(value)
                    for key, value in test_score.items():
                        test_score_history[key].append(value)
            else:
                print("\r%d" % (epoch+1), end="")

        if (return_score):
            return train_score_history, test_score_history
            

    def evaluate(
        self,
        n_train_eval = 1,
        n_test_eval = 0
    ):
        self.agent.eval()

        metrics = self.env.score([]).keys()
        train_score = { metric: [] for metric in metrics }
        test_score = { metric: [] for metric in metrics }
        
        # evaluate (train)
        for _ in range(n_train_eval):
            history, info_history = self.agent.interact_with(
                self.env,
                n_times = 1,
                phase = PhaseType.TRAINING,
                use_info = True
            )
            score = self.env.score(
                history,
                info_history
            )
            for key, value in score.items():
                train_score[key].append(value)
        
        # evlauate (test)
        for _ in range(n_test_eval):
            history, info_history = self.agent.interact_with(
                self.env,
                n_times = 1,
                phase = PhaseType.TEST,
                use_info = True
            )
            score = self.env.score(
                history,
                info_history
            )
            for key, value in score.items():
                test_score[key].append(value)

        return (train_score, test_score)

    def report(
        self,
        train_score,
        test_score
    ):
        print(train_score)
        print(test_score)
