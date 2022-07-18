#### DataLoader (Base) ####

from abc import ABC, abstractmethod


class DataLoaderBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def setup(self): raise NotImplementedError
    # @abstractmethod
    # def reset(self): raise NotImplementedError
    @abstractmethod
    def __iter__(self): raise NotImplementedError
    @abstractmethod
    def add(self): raise NotImplementedError
    @abstractmethod
    def save(self): raise NotImplementedError

    @property
    @abstractmethod
    def dataset(self): raise NotImplementedError
    @property
    @abstractmethod
    def dataloader(self): raise NotImplementedError
    @property
    @abstractmethod
    def batch_size(self): raise NotImplementedError
    @property
    @abstractmethod
    def shuffle(self): raise NotImplementedError
    @property
    @abstractmethod
    def num_workers(self): raise NotImplementedError
