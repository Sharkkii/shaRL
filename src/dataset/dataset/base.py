#### Dataset (Base) ####

from abc import ABC, abstractmethod

from ...common.data import SARS
from ...common.data import SGASG


class MemoryBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __len__(self): raise NotImplementedError
    @abstractmethod
    def __getitem__(self): raise NotImplementedError
    @abstractmethod
    def add_collection(self): raise NotImplementedError
    @abstractmethod
    def add_item(self): raise NotImplementedError
    @abstractmethod
    def add(self): raise NotImplementedError
    @abstractmethod
    def remove_collection(self): raise NotImplementedError
    @abstractmethod
    def remove_item(self): raise NotImplementedError
    @abstractmethod
    def remove(self): raise NotImplementedError

    @property
    @abstractmethod
    def collection(self): raise NotImplementedError
    @property
    @abstractmethod
    def transform(self): raise NotImplementedError
    @property
    @abstractmethod
    def size(self): raise NotImplementedError
    @property
    @abstractmethod
    def max_size(self): raise NotImplementedError


DatasetBase = MemoryBase


class StepwiseMemoryBase(MemoryBase):
    pass


class EpisodewiseMemoryBase(MemoryBase):
    pass
