#### Policy ####

import numpy as np
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

# TODO: should be moved to another file
def identity(*args):
    if (len(args) == 0):
        return None
    if (len(args) == 1):
        return args[0]
    else:
        return args
    return x


# base
class Policy(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        policy_network=None,
        value_network=None,
        qvalue_network=None,
        encoder=identity,
        decoder=identity
    ):
        super().__init__()
        self.policy_network = policy_network
        self.value_network = value_network
        self.qvalue_network = qvalue_network
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        state,
        action
    ):
        raise NotImplementedError


# default
class DefaultPolicy(Policy):

    def __init__(
        self,
        policy_network=None,
        value_network=None,
        qvalue_network=None,
        encoder=identity,
        decoder=identity
    ):
        assert(policy_network is not None)
        super().__init__(policy_network, encoder=encoder, decoder=decoder)

    def reset(self):
        pass

    def setup(self):
        pass
    
    def forward(self, state, action=None):
        if (action is None):
            x = torch.tensor(state, dtype=torch.float32)
        else:
            x = torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)
        x = self.encoder(x)
        x = self.policy_network(x)
        x = self.decoder(x)
        return x


# value-based policy
class ValueBasedPolicy(Policy):
    def __init__(
        self,
        policy_network=None,
        qvalue_network=None,
        value_network=None,
        encoder=identity,
        decoder=identity
    ):
        assert(qvalue_network is not None)
        super().__init__(qvalue_network=qvalue_network)
        self.setup(
            decoder=EpsilonGreedyDecoder()
        )
    
    def reset(self):
        pass

    def setup(
        self,
        encoder=None,
        decoder=None
    ):
        if (encoder is not None):
            self.encoder = encoder
        if (decoder is not None):
            self.decoder = decoder

    def forward(self, state, action=None):
        with torch.no_grad():
            if (action is None):
                x = torch.tensor(state, dtype=torch.float32)
            else:
                x = torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)
            x = self.encoder(x)
            x = self.qvalue_network(x)
            x = self.decoder(x)
        return x


# epsion-greedy policy
class EpsilonGreedyDecoder:

    def __init__(self, eps=0.0):
        assert(0 <= eps <= 1.0)
        self.eps = eps

    def __call__(self, x):
        with torch.no_grad():
            size = x.shape
            _, idx_argmax = torch.max(x, axis=0)
            # _, idx_argmax = torch.max(x, axis=1)
            x = torch.full(size=size, fill_value=self.eps)
            x[idx_argmax] += (1.0 - self.eps * len(x))
        return x
        # x_shape = x.shape[0]
        # x = np.eye(x_shape)[np.argmax(x, axis=1)]
        # return x
