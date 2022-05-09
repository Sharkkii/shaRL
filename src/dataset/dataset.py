#### Dataset ####

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


# T_STATE = torch.tensor
# T_ACTION = int
# T_REWARD = float


class BaseDataset(TorchDataset, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        collection = None,
        transform = None
    ):
        self.collection = None      
        self.transform = None
        self._is_available = False
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
        if (self.check_whether_valid_collection(collection)):
            self.collection = collection
            self._become_available()
        if (self.check_whether_valid_transform(transform)):
            self.transform = transform

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

    def __len__(
        self
    ):
        if (self.is_available):
            return len(self.collection)
        else:
            return 0
    
    def __getitem__(
        self,
        index
    ):
        index = index % len(self)
        item = self.collection[index]
        if (self.check_whether_valid_transform(self.transform)):
            item = self.transform(item)
        return item
