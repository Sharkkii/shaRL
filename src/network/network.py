#### Network ####

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

from .base import BaseNetwork
from .meta_network import MetaNetwork


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

class DefaultNetwork(BaseNetwork, nn.Module):
    
    def __init__(
        self,
        interface = None
    ):
        BaseNetwork.__init__(self, interface)
        nn.Module.__init__(self)
        self.l1 = nn.Linear(self.interface.sin[0], 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, self.interface.sout[0])
        self.reset()

    def reset(self):
        nn.init.normal_(self.l1.weight, mean=0., std=1.0)
        nn.init.normal_(self.l2.weight, mean=0., std=1.0)
        nn.init.normal_(self.l3.weight, mean=0., std=1.0)
        
    def forward(self, x):
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = self.l3(x)
        return x

class PseudoNetwork(BaseNetwork, nn.Module):
    
    def __init__(
        self,
        interface = None
    ):
        BaseNetwork.__init__(
            self,
            interface = interface
        )
        nn.Module.__init__(
            self
        )
        self.pseudo_parameters = nn.Parameter(torch.zeros(1))

    def reset(self):
        pass

    def forward(self, x):
        return x
