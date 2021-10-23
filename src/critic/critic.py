#### Critic ####

import os
import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from value import Value, QValue


class BaseCritic(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value = None,
        qvalue = None,
        # smooth_v = 0.99,
        # smooth_q = 0.99,
    ):
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        value_network = None,
        qvalue_network = None,
        value_optimizer = None,
        qvalue_optimizer = None
    ):
        self.value.setup(
            value_network = value_network,
            value_optimizer = value_optimizer
        )
        self.qvalue.setup(
            qvalue_network = qvalue_network,
            qvalue_optimizer = qvalue_optimizer
        )

    @abstractmethod
    def update_value(
        self,
        actor,
        trajectory
    ):
        raise NotImplementedError

    @abstractmethod
    def update_qvalue(
        self,
        actor,
        trajectory
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_value(
        self,
        actor,
        trajectory
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_qvalue(
        self,
        actor,
        trajectory
    ):
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        actor,
        trajectory,
        n_times = 1
    ):
        raise NotImplementedError

class Critic(BaseCritic):

    def __init__(
        self,
        value = None,
        qvalue = None,
        # smooth_v = 0.99,
        # smooth_q = 0.99,
    ):
        self.value = Value() if value is None else value
        self.qvalue = QValue() if qvalue is None else qvalue
        self.target_value = self.value.copy()
        self.target_qvalue = self.qvalue.copy()
        # self.smooth_v = smooth_v
        # self.smooth_q = smooth_q
    
    def reset(
        self
    ):
        pass

    def setup(
        self,
        value_network = None,
        qvalue_network = None,
        value_optimizer = None,
        qvalue_optimizer = None
    ):
        super().setup(
            value_network = value_network,
            qvalue_network = qvalue_network,
            value_optimizer = value_optimizer,
            qvalue_optimizer = qvalue_optimizer
        )

    def update_value(
        self,
        actor,
        trajectory
    ):
        pass

    def update_qvalue(
        self,
        actor,
        trajectory
    ):
        pass

    def update_target_value(
        self,
        actor,
        trajectory
    ):
        pass

    def update_target_qvalue(
        self,
        actor,
        trajectory
    ):
        pass

    def update(
        self,
        actor,
        trajectory,
        n_times = 1
    ):
        for _ in range(n_times):
            self.update_value(actor, trajectory)
            self.update_qvalue(actor, trajectory)
            self.update_target_value(actor, trajectory)
            self.update_target_qvalue(actor, trajectory)
