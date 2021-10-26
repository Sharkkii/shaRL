#### Network ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn


class BaseNetwork(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        network
    ):
        self.network = network if callable(network) else (lambda x: None)

    @abstractmethod
    def __call__(
        self,
        x
    ):
        return self.network(x)
    
    def parameters(
        self
    ):
        return self.network.parameters()
    
class ValueNetwork(BaseNetwork):

    def __init__(
        self,
        value_network
    ):
        self.network = value_network if callable(value_network) else (lambda state: None)
        # self.value_network = value_network if callable(value_network) else (lambda state: None)
    
    def __call__(
        self,
        state
    ):
        return self.network(state)
        # return self.value_network(state)

class DiscreteQValueNetwork(BaseNetwork):

    def __init__(
        self,
        qvalue_network
    ):
        self.network = qvalue_network if callable(qvalue_network) else (lambda state: None)
        # self.qvalue_network = qvalue_network if callable(qvalue_network) else (lambda state: None)
    
    def __call__(
        self,
        state,
        action = None
    ):
        assert(action is None)
        return self.network(state)
        # return self.qvalue_network(state)

class ContinuousQValueNetwork(BaseNetwork):

    def __init__(
        self,
        qvalue_network
    ):
        self.network = qvalue_network if callable(qvalue_network) else (lambda state, action: None)
        # self.qvalue_network = qvalue_network if callable(qvalue_network) else (lambda state, action: None)
    
    def __call__(
        self,
        state,
        action
    ):
        return self.network(state, action)
        # return self.qvalue_network(state, action)

QValueNetwork = DiscreteQValueNetwork

class PolicyNetwork(BaseNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state: None)
    
    def __call__(
        self,
        state
    ):
        return self.network(state)

    def predict(
        self,
        state
    ):
        return self.network.predict(state)

class VNet(nn.Module):
    
    def __init__(self, input_shape=4):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = 1
        self.l1 = nn.Linear(self.input_shape, 20)
        self.l2 = nn.Linear(20, 10)
        self.l3 = nn.Linear(10, self.output_shape)
        nn.init.normal_(self.l1.weight, mean=0., std=1.)
        nn.init.normal_(self.l2.weight, mean=0., std=1.)
        nn.init.normal_(self.l3.weight, mean=0., std=1.)
        
    def forward(self, x):
        x = torch.Tensor(x)
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = self.l3(x)
        return x

class QNet(nn.Module):

    def __init__(self, input_shape=4, output_shape=2):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, 20)
        self.l2 = nn.Linear(20, 10)
        self.l3 = nn.Linear(10, self.output_shape)
        nn.init.normal_(self.l1.weight, mean=0., std=1.)
        nn.init.normal_(self.l2.weight, mean=0., std=1.)
        nn.init.normal_(self.l3.weight, mean=0., std=1.)

    def forward(self, x, y=None):
        if (y is None):
            x = nn.ReLU()(self.l1(x))
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x
        else:
            x = torch.cat([x, y], dim=1)
            x = nn.ReLU()(self.l1(x))
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x

class PiNet(nn.Module):

    def __init__(self, input_shape=4, output_shape=2):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, 20)
        self.l2 = nn.Linear(20, 10)
        self.l3 = nn.Linear(10, self.output_shape)
        nn.init.normal_(self.l1.weight, mean=0., std=1.)
        nn.init.normal_(self.l2.weight, mean=0., std=1.)
        nn.init.normal_(self.l3.weight, mean=0., std=1.)
        
    def forward(self, x, y=None):
        if (y is None):
            x = nn.ReLU()(self.l1(x))
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x
        else:
            x = torch.cat([x, y], dim=1)
            x = nn.ReLU()(self.l1(x))
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x
    
    def predict(self, x):
        dim = x.ndim-1
        x = self.forward(x)
        x = nn.Softmax(dim=dim)(x)
        return x
