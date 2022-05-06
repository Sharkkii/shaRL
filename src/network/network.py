#### Network ####

from abc import ABCMeta, abstractmethod
import os
import torch
import torch.nn as nn

from .meta_network import MetaNetwork


class BaseNetwork(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self
    ):
        super().__init__()

    @abstractmethod
    def reset(
        self
    ):
        pass

class VNet(BaseNetwork):
    
    def __init__(self, input_shape=4):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = 1
        self.l1 = nn.Linear(self.input_shape, 20)
        # self.bn1 = nn.BatchNorm1d(20)
        self.l2 = nn.Linear(20, 20)
        # self.bn2 = nn.BatchNorm1d(20)
        self.l3 = nn.Linear(20, self.output_shape)
        self.reset()

    def reset(self):
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

class QNet(BaseNetwork):

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
        self.reset()
    
    def reset(self):
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

class PiNet(BaseNetwork):

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
        self.reset()
    
    def reset(self):
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

class PseudoNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.pseudo_parameters = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x
