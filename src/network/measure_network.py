#### Network ####

from abc import ABCMeta, abstractmethod
import os
import torch
import torch.nn as nn

from ..common import Component

class BaseMeasureNetwork(Component, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        network = None
    ):
        Component.__init__(self)
        self.network = None
        self.setup(network)

    @abstractmethod
    def __call__(
        self,
        **x
    ):
        return self.network(**x)

    @abstractmethod
    def reset(
        self
    ):
        pass

    @abstractmethod
    def setup(
        self,
        network = None
    ):
        if (network is not None):
            assert(callable(network))
            self.network = network
            self._become_available()

    def train(
        self
    ):
        self.network.train()

    def eval(
        self
    ):
        self.network.eval()
    
    @Component.check_whether_available
    def parameters(
        self
    ):
        return self.network.parameters()

    def save(
        self,
        path_to_network
    ):
        ext = ".pth"
        path_to_network = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", path_to_network)) + ext
        torch.save(self.network.state_dict(), path_to_network)

    def load(
        self,
        path_to_network
    ):
        ext = ".pth"
        path_to_network = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", path_to_network)) + ext
        self.network.load_state_dict(torch.load(path_to_network))

class BasePolicyNetwork(BaseMeasureNetwork, metaclass=ABCMeta):

    @abstractmethod
    def P(
        self,
        state,
        action
    ):
        raise NotImplementedError

    @abstractmethod
    def logP(
        self,
        state,
        action
    ):
        raise NotImplementedError

class PseudoMeasureNetwork(BaseMeasureNetwork):

    @classmethod
    def __raise_exception(cls):
        raise Exception("`PseudoMeasureNetwork` cannot be used as a measure network.")

    def __init__(
        self,
        network = None
    ):
        pass
    
    def reset(self):
        pass

    def setup(
        self,
        network = None
    ):
        pass

    def train(
        self
    ):
        pass

    def eval(
        self
    ):
        pass

    def __call__(
        self,
        x
    ):
        PseudoMeasureNetwork.__raise_exception()

    def parameters(
        self
    ):
        pseudo_parameters = (torch.zeros(1),)
        return pseudo_parameters
    
    def save(
        self,
        path_to_network
    ):
        PseudoMeasureNetwork.__raise_exception()

    def load(
        self,
        path_to_network
    ):
        PseudoMeasureNetwork.__raise_exception()
