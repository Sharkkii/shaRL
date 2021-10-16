#### Critic ####

import sys
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F


class BaseCritic(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value = None,
        qvalue = None,
        # smooth_v = 0.99,
        # smooth_q = 0.99,
    ):
        self.value = value
        self.qvalue = qvalue
        self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        # self.smooth_v = smooth_v
        # self.smooth_q = smooth_q
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def update_value(
        self,
        trajs
    ):
        raise NotImplementedError

    @abstractmethod
    def update_qvalue(
        self,
        trajs
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_value(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def update_target_qvalue(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
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
        self.value = value
        self.qvalue = qvalue
        self.target_value = value.copy()
        self.target_qvalue = qvalue.copy()
        # self.smooth_v = smooth_v
        # self.smooth_q = smooth_q
    
    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass

    def update_value(
        self,
        trajectory
    ):
        pass

    def update_qvalue(
        self,
        trajectory
    ):
        pass

    def update_target_value(
        self,
        trajectory
    ):
        pass

    def update_target_qvalue(
        self,
        trajectory
    ):
        pass

    def update(
        self,
        trajectory,
        n_times = 1
    ):
        for _ in range(n_times):
            self.update_value(trajectory)
            self.update_qvalue(trajectory)
            self.update_target_value(trajectory)
            self.update_target_qvalue(trajectory)
