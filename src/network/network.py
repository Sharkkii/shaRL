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

class VNet(nn.Module):
    
    def __init__(self, input_shape=4):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = 1
        self.l1 = nn.Linear(self.input_shape, 10)
        self.l2 = nn.Linear(10, 10)
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

    def __init__(
        self,
        input_shape=4,
        output_shape=2
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, self.output_shape)
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
            x = nn.ReLU()(self.l2(x))
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
        self.l1 = nn.Linear(self.input_shape, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, self.output_shape)
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
            x = nn.ReLU()(self.l2(x))
            x = self.l3(x)
            return x
    
    def predict(self, x):
        dim = x.ndim-1
        x = self.forward(x)
        x = nn.Softmax(dim=dim)(x)
        return x
