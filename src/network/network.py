#### Network ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn


class BaseNetwork(metaclass=ABCMeta):

    def __init__(
        self,
        network
    ):
        self.network = network
    
    def __call__(
        self,
        x
    ):
        return self.network(x)
    
class ValueNetwork(BaseNetwork):

    def __init__(
        self,
        value_network
    ):
        self.value_network = value_network
    
    def __call__(
        self,
        state
    ):
        return self.value_network(state)

class QValueNetwork(BaseNetwork):

    def __init__(
        self,
        qvalue_network
    ):
        self.qvalue_network = qvalue_network
    
    def __call__(
        self,
        state,
        action
    ):
        return self.qvalue_network(state, action)

class PolicyNetwork(BaseNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.policy_network = policy_network
    
    def __call__(
        self,
        state,
        action = None
    ):
        return self.policy_network(state, action)

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
        
    def forward(self, x):
        dim = x.ndim-1
        x = torch.Tensor(x)
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = nn.Softmax(dim=dim)(self.l3(x))
        return x
