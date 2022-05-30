#### Controller ####

import warnings
from abc import ABCMeta, abstractmethod

from ..const import PhaseType
from ..const import EnvironmentModelType, AgentStrategyType, AgentBehaviorType, AgentLearningType
from ..dataset import SarsDataset
from ..dataset import SagsDataset
from ..dataset import DataLoader


class BaseController(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(
        self,
        environment = None,
        agent = None,
        config_e = None,
        config_a = None
    ):
        self.env = environment
        self.agent = agent
        self._is_available = False
        self.setup(
            environment = environment,
            agent = agent,
            config_e = config_e,
            config_a = config_a
        )
    
    @abstractmethod
    def setup(
        self,
        environment = None,
        agent = None,
        config_e = None,
        config_a = None
    ):
        if ((environment is not None) and (agent is not None)):
            
            self.env = environment
            self.agent = agent
            
            if (config_e is None):
                self.env.setup()
            else:
                self.env.setup(config_e)

            if (config_a is None):
                self.agent.setup()
            else:
                self.agent.setup(config_a)

            self._become_available()
    
    @abstractmethod
    def reset(
        self
    ):
        if ((self.env is not None) and (self.agent is not None)):
            self.env.reset()
            self.agent.reset()

    @property
    def is_available(
        self
    ):
        return self._is_available

    def _become_available(
        self
    ):
        self._is_available = True

    def _become_unavailable(
        self
    ):
        self._is_available = False

class RLController(BaseController):

    def __init__(
        self,
        environment = None,
        agent = None,
        config_e = None,
        config_a = None
    ):
        super().__init__(
            environment = environment,
            agent = agent,
            config_e = config_e,
            config_a = config_a
        )
    
    def setup(
        self,
        environment = None,
        agent = None,
        config_e = None,
        config_a = None
    ):
        super().setup(
            environment = environment,
            agent = agent,
            config_e = config_e,
            config_a = config_a
        )

    def reset(
        self
    ):
        super().reset()

    # alias of `train` (for compatibility)
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
        gradient_step = 1,
        dataset_size = 100000,
        batch_size = 100,
        shuffle = True,
        return_score = True,
    ):
        return self.train(
            self,
            n_epoch = n_epoch,
            n_sample = n_sample,
            n_sample_start = n_sample_start,
            n_eval = n_eval, # deprecated
            n_train_eval = n_train_eval,
            n_test_eval = n_test_eval,
            n_eval_interval = n_eval_interval,
            n_env_step = env_step,
            n_gradient_step = gradient_step,
            max_dataset_size = dataset_size,
            batch_size = batch_size,
            shuffle = shuffle,
            return_score = return_score,
        )
    
    def train(
        self,
        n_epoch = 1,
        n_sample = 1,
        n_sample_start = 1,
        n_eval = None, # deprecated
        n_train_eval = 1,
        n_test_eval = 0,
        n_eval_interval = 10,
        n_env_step = 1,
        n_gradient_step = 1,
        max_dataset_size = 100000,
        batch_size = 100,
        shuffle = True,
        return_score = True,
    ):

        if (n_gradient_step < 0):
            raise ValueError("`n_gradient_step` must be greater than 0.")

        if (n_eval is not None):
            warnings.warn("`n_eval` is deprecated. Use `n_train_eval` & `n_test_eval` instead.")
            n_train_eval = n_eval
            n_test_eval = 0

        model_type = EnvironmentModelType.MODEL_FREE
        strategy_type = AgentStrategyType.VALUE_BASED
        behavior_type = AgentBehaviorType.OFF_POLICY
        learning_type = AgentLearningType.OFFLINE

        # assert(n_sample <= n_sample_start <= self.agent.memory.capacity)

        dataset = SarsDataset(
            collection = [],
            transform = None
        )
        dataloader = DataLoader(
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

            for step in range(n_env_step):

                self.agent.setup_on_every_step(
                    step = step,
                    n_step = n_env_step
                )

                # interact w/ environment
                if (model_type == EnvironmentModelType.MODEL_FREE):
                    trajs_env = self.agent.interact_with_env(
                        env = self.env,
                        n_episode = 1,
                        information = {
                            "phase": PhaseType.TRAINING
                        }
                    )

                # interact w/ model    
                if (model_type == EnvironmentModelType.MODEL_BASED):
                    trajs_model = self.agent.interact_with_env(
                        env = self.agent.model,
                        n_episode = 1,
                        information = {
                            "phase": PhaseType.TRAINING
                        }
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
            for step in range(n_gradient_step):

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
            history, info_history = self.agent.interact_with_env(
                self.env,
                n_episode = 1,
                information = {
                    "phase": PhaseType.TRAINING
                },
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
            history, info_history = self.agent.interact_with_env(
                self.env,
                n_episode = 1,
                information = {
                    "phase": PhaseType.TEST
                },
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


Controller = RLController


class GoalConditionedRLController(BaseController):

    def __init__(
        self,
        environment = None,
        agent = None,
        config_e = None,
        config_a = None
    ):
        super().__init__(
            environment = environment,
            agent = agent,
            config_e = config_e,
            config_a = config_a
        )
    
    def setup(
        self,
        environment = None,
        agent = None,
        config_e = None,
        config_a = None
    ):
        super().setup(
            environment = environment,
            agent = agent,
            config_e = config_e,
            config_a = config_a
        )

    def reset(
        self
    ):
        super().reset()
    
    def train(
        self,
        n_epoch = 1,
        # n_sample = 1,
        # n_sample_start = 1,
        n_train_eval = 1,
        n_test_eval = 0,
        n_eval_interval = 10,
        n_env_step = 1,
        n_gradient_step = 1,
        max_dataset_size = 100000,
        batch_size = 100,
        shuffle = True,
        return_score = True,
    ):

        if (n_gradient_step < 0):
            raise ValueError("`n_gradient_step` must be greater than 0.")

        model_type = EnvironmentModelType.MODEL_FREE
        strategy_type = AgentStrategyType.VALUE_BASED
        behavior_type = AgentBehaviorType.OFF_POLICY
        learning_type = AgentLearningType.OFFLINE

        dataset = SagsDataset(
            collection = [],
            transform = None
        )
        dataloader = DataLoader(
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

            history_env = []
            history_model = []

            for step in range(n_env_step):

                self.agent.setup_on_every_step(
                    step = step,
                    n_step = n_env_step
                )

                if (model_type == EnvironmentModelType.MODEL_FREE):
                    history_env = self.agent.interact_with_env(
                        env = self.env,
                        n_episode = 1,
                        information = {
                            "phase": PhaseType.TRAINING
                        }
                    )

                if (model_type == EnvironmentModelType.MODEL_BASED):
                    history_model = self.agent.interact_with_env(
                        env = self.agent.model,
                        n_episode = 1,
                        information = {
                            "phase": PhaseType.TRAINING
                        }
                    )

            history = history_env + history_model
            dataloader.save(history)
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

            for step in range(n_gradient_step):

                # history: {(s,a,g)}
                for history in dataloader:
                    # guard
                    if (len(history[0]) < batch_size):
                        continue
                    self.agent.update_critic(history, n_times=1)
                    self.agent.update_actor(history, n_times=1)

                    # learn dynamics
                    # if (model_type == EnvironmentModelType.MODEL_BASED):
                    #     self.agent.update_model(history, n_times=1)

            # learn something if needed
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
            history, info_history = self.agent.interact_with_env(
                self.env,
                n_episode = 1,
                information = {
                    "phase": PhaseType.TRAINING
                },
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
            history, info_history = self.agent.interact_with_env(
                self.env,
                n_episode = 1,
                information = {
                    "phase": PhaseType.TEST
                },
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
