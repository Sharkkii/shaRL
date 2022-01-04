#### Network ####

from abc import ABCMeta, abstractmethod
import os
import torch
import torch.nn as nn

from .meta_network import MetaNetwork

class BaseNetwork(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        network
    ):
        assert(callable(network))
        self.network = network

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
        **kwargs
    ):
        pass

    def train(
        self
    ):
        self.network.train()

    def eval(
        self
    ):
        self.network.eval()
    
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

class BasePolicyNetwork(BaseNetwork, metaclass=ABCMeta):

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

class VNet(nn.Module):
    
    def __init__(self, input_shape=4):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = 1
        self.l1 = nn.Linear(self.input_shape, 20)
        # self.bn1 = nn.BatchNorm1d(20)
        self.l2 = nn.Linear(20, 20)
        # self.bn2 = nn.BatchNorm1d(20)
        self.l3 = nn.Linear(20, self.output_shape)
        nn.init.normal_(self.l1.weight, mean=0., std=1.0)
        nn.init.normal_(self.l2.weight, mean=0., std=1.0)
        nn.init.normal_(self.l3.weight, mean=0., std=1.0)
        
    def forward(self, x):
        x = nn.ReLU()(self.l1(x))
        # x = nn.ReLU()(self.bn1(self.l1(x)))
        x = nn.ReLU()(self.l2(x))
        # x = nn.ReLU()(self.bn2(self.l2(x)))
        x = self.l3(x)
        return x

class QNet(nn.Module):

    def __init__(
        self,
        input_shape=4,
        output_shape=2
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, 20)
        # self.bn1 = nn.BatchNorm1d(20)
        self.l2 = nn.Linear(20, 20)
        # self.bn2 = nn.BatchNorm1d(20)
        self.l3 = nn.Linear(20, self.output_shape)
        nn.init.normal_(self.l1.weight, mean=0., std=1.0)
        nn.init.normal_(self.l2.weight, mean=0., std=1.0)
        nn.init.normal_(self.l3.weight, mean=0., std=1.0)

    def forward(self, x, y=None):
        if (y is None):
            x = nn.ReLU()(self.l1(x))
            # x = nn.ReLU()(self.bn1(self.l1(x)))
            x = nn.ReLU()(self.l2(x))
            # x = nn.ReLU()(self.bn2(self.l2(x)))
            x = self.l3(x)
            return x
        else:
            x = torch.cat([x, y], dim=1)
            x = nn.ReLU()(self.l1(x))
            # x = nn.ReLU()(self.bn1(self.l1(x)))
            x = nn.ReLU()(self.l2(x))
            # x = nn.ReLU()(self.bn2(self.l2(x)))
            x = self.l3(x)
            return x

class PiNet(nn.Module):

    def __init__(
        self,
        input_shape = 4,
        output_shape = 2
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, 20)
        # self.bn1 = nn.BatchNorm1d(20)
        self.l2 = nn.Linear(20, 20)
        # self.bn2 = nn.BatchNorm1d(20)
        self.l3 = nn.Linear(20, self.output_shape)
        nn.init.normal_(self.l1.weight, mean=0., std=1.0)
        nn.init.normal_(self.l2.weight, mean=0., std=1.0)
        nn.init.normal_(self.l3.weight, mean=0., std=1.0)
        
    def forward(self, x, y=None):
        if (y is None):
            x = nn.ReLU()(self.l1(x))
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x
        else:
            x = torch.cat([x, y], dim=1)
            x = nn.ReLU()(self.l1(x))
            # x = nn.ReLU()(self.bn1(self.l1(x)))
            x = nn.ReLU()(self.l2(x))
            # x = nn.ReLU()(self.bn2(self.l2(x)))
            x = self.l3(x)
            return x
    
    # def predict(self, x):
    #     dim = x.ndim-1
    #     x = self.forward(x)
    #     x = nn.Softmax(dim=dim)(x)
    #     return x

class DefaultNetwork(nn.Module, metaclass=MetaNetwork):
    spec = "default"
