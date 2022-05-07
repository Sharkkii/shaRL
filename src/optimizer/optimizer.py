#### Optimizer ####

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.optim

from ..network import BaseMeasureNetwork
from ..network import PseudoMeasureNetwork

class BaseMeasureOptimizer(metaclass=ABCMeta):

    def check_whether_available(f):
        def wrapper(self, *args, **kwargs):
            if (self.is_available):
                raise Exception(f"Call `Optimizer.setup` before using `Optimizer.{ f.__name__ }`")
            return f(self, *args, **kwargs)
        return wrapper

    @abstractmethod
    def __init__(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def reset(
        self
    ):
        raise NotImplementedError
    
    @abstractmethod
    def setup(
        self
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

    @check_whether_available
    def zero_grad(
        self
    ):
        self.optimizer.zero_grad()

    @check_whether_available
    def step(
        self
    ):
        self.optimizer.step()

    @check_whether_available
    def clip_grad_value(
        self,
        value = 1.0
    ):
        for parameter in self.network.parameters():
            torch.nn.utils.clip_grad_value_(parameter.grad, value)
    
    @check_whether_available
    def clip_grad_norm(
        self,
        value = 1.0
    ):
        for parameter in self.network.parameters():
            torch.nn.utils.clip_grad_norm_(parameter.grad, value, norm_type=2)

class TemplateMeasureOptimizer(BaseMeasureOptimizer):

    # How to define `MeasureOptimizer`:
    #
    # class MeasureOptimizer(TemplateMeasureOptimizer):
    #     factory = ****
    #

    def __init__(
        self,
        network = None,
        **kwargs
    ):
        self.optimizer = None
        self._is_available = False

        self.setup(
            network = network,
            **kwargs
        )

    def reset(
        self
    ):
        pass
    
    def setup(
        self,
        network,
        **kwargs
    ):
        if (isinstance(network, BaseMeasureNetwork)):
            self.optimizer = self.factory(
                network.parameters(),
                **kwargs
            )
            self.network = network
            self._become_available()
            print(f"Optimizer.setup: { self.factory }({ network })")

class MetaMeasureOptimizer(ABCMeta, type):

    def __new__(
        cls,
        name,
        bases,
        namespace
    ):
        assert("factory" in namespace)
        bases = bases + (BaseMeasureOptimizer,)
        namespace["__metaclass__"] = (MetaMeasureOptimizer,)
        namespace["__init__"] = TemplateMeasureOptimizer.__init__
        namespace["reset"] = TemplateMeasureOptimizer.reset
        namespace["setup"] = TemplateMeasureOptimizer.setup
        namespace["zero_grad"] = TemplateMeasureOptimizer.zero_grad
        namespace["step"] = TemplateMeasureOptimizer.step
        return super().__new__(cls, name, bases, namespace)

class MeasureOptimizer(metaclass=MetaMeasureOptimizer):
    factory = torch.optim.Adam

# will be deprecated (replaced by `BaseMeasureOptimizer`)
BaseOptimizer = BaseMeasureOptimizer

# will be deprecated (replaced by `MeasureOptimizer`)
class Optimizer(BaseOptimizer):

    def check_whether_available(f):
        def wrapper(self, *args, **kwargs):
            if (self.optimizer is None):
                raise Exception(f"Call `Optimizer.setup` before using `Optimizer.{ f.__name__ }`")
            return f(self, *args, **kwargs)
        return wrapper

    def __init__(
        self,
        optimizer,
        network = None,
        **kwargs
    ):
        self.optimizer_factory = optimizer # should be implemented as MetaClass
        self.optimizer = None
        self.network = None
        self._is_available = False

        self.setup(
            network = network,
            **kwargs
        )

    def reset(
        self
    ):
        raise NotImplementedError
    
    @check_whether_available
    def zero_grad(
        self
    ):
        self.optimizer.zero_grad()
    
    def setup(
        self,
        network,
        **kwargs
    ):
        if (isinstance(network, BaseMeasureNetwork)):
            self.optimizer = self.optimizer_factory(
                network.parameters(),
                **kwargs
            )
            self.network = network
            self._become_available()
            print(f"Optimizer.setup: { self.optimizer_factory }({ network })")

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

    @check_whether_available
    def step(
        self
    ):
        self.optimizer.step()

    @check_whether_available
    def clip_grad_value(
        self,
        value = 1.0
    ):
        for parameter in self.network.parameters():
            torch.nn.utils.clip_grad_value_(parameter.grad, value)
    
    @check_whether_available
    def clip_grad_norm(
        self,
        value = 1.0
    ):
        for parameter in self.network.parameters():
            torch.nn.utils.clip_grad_norm_(parameter.grad, value, norm_type=2)
