#### Critic ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F

from ..value import Value
from ..value import PseudoValue
from ..value import QValue
from ..value import PseudoQValue
from ..value import cast_to_value
from ..value import cast_to_qvalue


class BaseCritic(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        value = None,
        qvalue = None,
        # smooth_v = 0.99,
        # smooth_q = 0.99,
    ):
        self.value = None # cast_to_value(value)
        self.qvalue = None # cast_to_qvalue(qvalue)
        self.target_value = None # self.value.copy()
        self.target_qvalue = None # self.qvalue.copy()
        self._is_available = False
        self.setup(
            value = value,
            qvalue = qvalue
        )
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        env = None, # will be deprecated
        value_network = None, # will be deprecated
        qvalue_network = None, # will be deprecated
        value_optimizer = None, # will be deprecated
        qvalue_optimizer = None, # will be deprecated
        value = None,
        qvalue = None
    ):
        if ((value is not None) and (qvalue is not None)):
            self.value = value
            self.qvalue = qvalue
            self.target_value = self.value.copy()
            self.target_qvalue = self.qvalue.copy()
            self._become_available()

    @abstractmethod
    def setup_with_actor(
        self,
        actor
    ):
        raise NotImplementedError
    
    @abstractmethod
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        raise NotImplementedError

    @abstractmethod
    def setup_on_every_step(
        self,
        step,
        n_step
    ):
        raise NotImplementedError

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

    def train(
        self
    ):
        self.value.train()
        self.qvalue.train()

    def eval(
        self
    ):
        self.value.eval()
        self.qvalue.eval()

    def save(
        self,
        path_to_value,
        path_to_qvalue,
    ):
        self.value.save(path_to_value)
        self.qvalue.save(path_to_qvalue)
    
    def load(
        self,
        path_to_value,
        path_to_qvalue,
    ):
        self.value.load(path_to_value)
        self.qvalue.load(path_to_qvalue)

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
        qvalue = None
    ):
        super().__init__(
            value = value,
            qvalue = qvalue
        )
    
    def reset(
        self
    ):
        pass

    def setup(
        self,
        env = None, # will be deprecated
        value_network = None, # will be deprecated
        qvalue_network = None, # will be deprecated
        value_optimizer = None, # will be deprecated
        qvalue_optimizer = None, # will be deprecated
        value = None,
        qvalue = None
    ):
        super().setup(
            value = value,
            qvalue = qvalue
        )
        # self.env = env
        # self.value.setup(
        #     value_network = value_network,
        #     value_optimizer = value_optimizer
        # )
        # self.qvalue.setup(
        #     qvalue_network = qvalue_network,
        #     qvalue_optimizer = qvalue_optimizer
        # )
        # self.target_value = self.value.copy()
        # self.target_qvalue = self.qvalue.copy()

    def setup_with_actor(
        self,
        actor
    ):
        pass
    
    def setup_on_every_epoch(
        self,
        epoch,
        n_epoch
    ):
        pass

    def setup_on_every_step(
        self,
        step,
        n_step
    ):
        pass

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
