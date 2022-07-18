#### Dataset (Mixin) ####

from .base import DatasetBase
from torch.utils.data import Dataset as TorchDataset

from ...common import Component


MAX_SIZE = 10000


class DatasetMixin(DatasetBase, Component, TorchDataset):

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
        Component.__init__(self)
        DatasetMixin.declare(self)
        DatasetMixin.setup(
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
            return

        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        
        self._collection = collection
        self._max_size = max_size
        if (self.check_whether_valid_transform(transform)):
            self._transform = transform
        self.trancate()

        self._become_available()

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
        return (collection is not None) and (hasattr(collection, "__iter__")) and (hasattr(collection, "__getitem__"))

    def check_whether_valid_transform(self, transform):
        return callable(transform)
