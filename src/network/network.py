#### Network ####

import numpy as np
import torch
import torch.nn as nn

# Value
class ValueNetwork(nn.Module):
    
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


# Q-value
class QvalueNetwork(nn.Module):

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


# Deterministic Policy
class DiscretePolicyNetwork(nn.Module):
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


# Continuous Policy
class ContinuousPolicyNetwork(nn.Module):

    def __init__(self, input_shape=3, output_shape=2, min=min, max=max):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.min = min
        self.max = max
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
        x = torch.clamp(self.l3(x))
        return x


PolicyNetwork = DiscretePolicyNetwork


# NOTE: see torch.nn.module
# NOTE: see torch.nn.Linear (for example)
class TabularNetwork(nn.Module):

    def __init__(
        self,
        input_shape=1,
        output_shape=1
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.table = nn.Parameter(torch.Tensor(self.input_shape, self.output_shape))
        nn.init.uniform_(self.table)

    def parameters(self):
        # yield self.table
        return [self.table]

    def __call__(self, x, y=None):
        return self.forward(x, y)
    
    @torch.no_grad()
    def forward(self, x, y=None):
        flag = (x.dim() == 1)
        if (flag): x = x[None, :]
        
        _, x = torch.max(x, dim=1)
        if (y is None):
            ret = self.table[x]
        else:
            ret = self.table[x][y]
        
        if (flag): ret = torch.squeeze(ret)
        return ret
