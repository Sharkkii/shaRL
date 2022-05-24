#### Dataset ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..common import Component
from .data import SARS


# T_STATE = torch.tensor
# T_ACTION = int
# T_REWARD = float


class BaseDataset(Component, TorchDataset, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        collection = None,
        transform = None
    ):
        Component.__init__(self)
        self.collection = None      
        self.transform = None
        self.setup(
            collection = collection,
            transform = transform
        )
    
    @abstractmethod
    def reset(
        self
    ):
        pass

    @abstractmethod
    def setup(
        self,
        collection = None,
        transform = None
    ):
        if (collection is None):
            return

        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        
        self.collection = collection
        if (self.check_whether_valid_transform(transform)):
            self.transform = transform
        self._become_available()

    @abstractmethod
    def __len__(
        self
    ):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(
        self,
        index
    ):
        raise NotImplementedError

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        self.collection.extend(collection)

    def add_item(
        self,
        item
    ):
        self.add_collection([ item ])

    def add(
        self,
        collection
    ):
        self.add_collection(collection)

    def remove_collection(
        self,
        n = 0
    ):
        if (n > 0):
            del self.collection[:n]
    
    def remove_item(
        self
    ):
        self.remove_collection(n = 1)

    def remove(
        self,
        n = 0
    ):
        self.remove_collection(n = n)

    def check_whether_valid_collection(self, collection):
        return (collection is not None) and (hasattr(collection, "__iter__")) and (hasattr(collection, "__getitem__"))

    def check_whether_valid_transform(self, transform):
        return callable(transform)


class Dataset(BaseDataset):

    def __init__(
        self,
        collection = None,
        transform = None
    ):
        super().__init__(
            collection = collection,
            transform = transform
        )

    def reset(
        self
    ):
        super().reset()

    def setup(
        self,
        collection = None,
        transform = None
    ):
        super().setup(
            collection = collection,
            transform = transform
        )

    @Component.check_whether_available
    def __len__(
        self
    ):
        if (len(self.collection) > 0):
            return len(self.collection)
        else:
            return 1
    
    @Component.check_whether_available
    def __getitem__(
        self,
        index
    ):
        if (len(self.collection) == 0):
            return None

        index = index % len(self)
        item = self.collection[index]
        if (self.check_whether_valid_transform(self.transform)):
            item = self.transform(item)
        return item


class SarsDataset(Dataset):

    def __init__(
        self,
        collection = None,
        transform = None
    ):
        super().__init__(
            collection = collection,
            transform = transform
        )

    def setup(
        self,
        collection = None,
        transform = None
    ):
        if (collection is None):
            return

        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")

        if (not self.check_whether_valid_sars_collection(collection)):
            raise ValueError("`collection` must be 'List[SARS]' object.")

        super().setup(
            collection = collection,
            transform = transform
        )

    def __getitem__(
        self,
        index
    ):
        sars = super().__getitem__(index = index)
        return (sars.state, sars.action, sars.reward, sars.next_state)

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")

        if (not self.check_whether_valid_sars_collection(collection)):
            raise ValueError("`collection` must be 'List[SARS]' object.")

        super().add_collection(collection)

    def check_whether_valid_sars_collection(self, collection):
        return all([ (type(item) is SARS) for item in collection ])
