#### DataLoader (Mixin) ####

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data._utils.collate import default_collate
from ...common import Component

from ..dataset import DatasetBase
from .base import DataLoaderBase


def custom_collate_fn(batch):
    if (all([(item is None) for item in batch])):
        return None
    return default_collate(batch)


class DataLoaderMixin(DataLoaderBase, Component):

    def declare(self):
        self._dataset = None
        self._dataloader = None
        self._batch_size = None
        self._shuffle = None
        self._num_workers = None

    @property
    def dataset(self): return self._dataset
    @property
    def dataloader(self): return self._dataloader
    @property
    def batch_size(self): return self._batch_size
    @property
    def shuffle(self): return self._shuffle
    @property
    def num_workers(self): return self._num_workers

    def __init__(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    ):
        DataLoaderMixin.declare(self)
        Component.__init__(self)
        self.setup(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )

    def setup(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    ):
        if (self.check_whether_valid_size(batch_size)):
            self._batch_size = batch_size
        if (self.check_whether_valid_flag(shuffle)):
            self._shuffle = shuffle

        if (
            self.check_whether_valid_size(self.batch_size) and
            self.check_whether_valid_flag(self.shuffle) and
            self.check_whether_valid_dataset(dataset)
        ):
            self._dataset = dataset
            self._dataloader = TorchDataLoader(
                dataset = dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                num_workers = num_workers,
                collate_fn = custom_collate_fn
            )
            self._become_available()

    def __iter__(
        self
    ):
        class _iterator:
            def __init__(_self): pass
            def __iter__(_self): return _self
            def __next__(_self): return None

        if (len(self.dataset.collection) == 0):
            return _iterator()
        return iter(self.dataloader)

    @Component.check_whether_available
    def add(
        self,
        collection
    ):
        self.dataset.add(collection = collection)

    @Component.check_whether_available
    def save(
        self,
        collection
    ):
        self.add(collection = collection)
    
    def check_whether_valid_size(self, size):
        return (type(size) is int) and (size > 0)

    def check_whether_valid_flag(self, flag):
        return (type(flag) is bool)

    def check_whether_valid_dataset(self, dataset):
        return (
            isinstance(dataset, DatasetBase) and
            dataset.is_available
        )
