#### Dataset (Base Class) ####

from abc import ABC, abstractmethod

from ...common.data import SARS
from ...common.data import SGASG


class MemoryBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    @abstractmethod
    def __len__(self): raise NotImplementedError
    @abstractmethod
    def __getitem__(self): raise NotImplementedError
    @abstractmethod
    def add(self): raise NotImplementedError
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
    def max_size(self): raise NotImplementedError
    @property
    @abstractmethod
    def size(self): raise NotImplementedError


DatasetBase = MemoryBase


class StepwiseMemoryBase(MemoryBase):
    pass


class EpisodeMemoryBase(MemoryBase):
    pass
