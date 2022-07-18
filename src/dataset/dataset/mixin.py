#### Dataset (Mixin) ####

from torch.utils.data import Dataset as TorchDataset
from ...common import Component

from .base import DatasetBase
from .base import MemoryBase
from .base import StepwiseMemoryBase
from .base import EpisodeMemoryBase


MAX_SIZE = 10000


class MemoryMixin(MemoryBase, TorchDataset, Component):

    def declare(self):
        self._collection = None
        self._transform = None
        self._max_size = None

    @property
    def collection(self): return self._collection
    @property
    def transform(self): return self._transform
    @property
    def max_size(self): return self._max_size
    @property
    def size(self): return len(self.collection)

    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE,
    ):
        MemoryMixin.declare(self)
        Component.__init__(self)
        MemoryMixin.setup(
            self,
            collection = collection,
            transform = transform,
            max_size = max_size
        )

    def setup(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE
    ):
        if (collection is None):
            collection = []
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        flag = False
        if (self.check_whether_valid_transform(transform)):
            flag = True
        if (not self.check_whether_valid_size(max_size)):
            raise ValueError("`max_size` must be 'Int'.")

        self._collection = collection
        if (flag): self._transform = transform
        self._max_size = max_size
        self.trancate()

        self._become_available()

    def __len__(
        self
    ):
        if (len(self.collection) == 0):
            return 1
        else:
            return len(self.collection)

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

    def add(
        self,
        collection
    ):
        self.add_collection(collection)

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        self.collection.extend(collection)
        self.trancate()

    def add_item(
        self,
        item
    ):
        self.add_collection([ item ])

    def remove(
        self,
        n = 0
    ):
        self.remove_collection(n = n)

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

    def trancate(
        self
    ):
        del self.collection[:-self.max_size]

    def check_whether_valid_collection(self, collection):
        return (
            (collection is not None) and
            (hasattr(collection, "__iter__")) and
            (hasattr(collection, "__getitem__"))
        )

    def check_whether_valid_transform(self, transform):
        return callable(transform)

    def check_whether_valid_size(self, size):
        return (type(size) is int) and (size > 0)


class StepwiseMemoryMixin(MemoryMixin, StepwiseMemoryBase):

    def check_whether_valid_collection(self, collection):
        if (collection is None): return False
        if (type(collection) is list):
            if (len(collection) == 0): return True
            return (type(collection[0]) is not list)
        else:
            return False


class EpisodeMemoryMixin(MemoryMixin, EpisodeMemoryBase):

    def check_whether_valid_collection(self, collection):
        if (collection is None): return False
        if (type(collection is list)):
            if (len(collection) == 0): return True
            if (type(collection[0]) is list):
                if (len(collection[0]) == 0): return True
                return (type(collection[0][0]) is not list)
        else:
            return False


DatasetMixin = MemoryMixin
