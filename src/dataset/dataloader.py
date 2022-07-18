#### DataLoader ####

from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data._utils.collate import default_collate

from ..common import Component
from .dataset import DatasetBase


def custom_collate_fn(batch):
    if (all([(item is None) for item in batch])):
        return None
    return default_collate(batch)


class BaseDataLoader(Component, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    ):
        Component.__init__(self)
        self.dataset = None
        self.dataloader = None
        self.batch_size = None
        self.shuffle = None
        self.num_workers = None
        self.setup(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )
        
    def reset(
        self
    ):
        pass

    def setup(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    ):
        if (self.check_whether_valid_size(batch_size)):
            self.batch_size = batch_size
        if (self.check_whether_valid_flag(shuffle)):
            self.shuffle = shuffle

        if (
            self.check_whether_valid_size(self.batch_size) and
            self.check_whether_valid_flag(self.shuffle) and
            self.check_whether_valid_dataset(dataset)
        ):
            self.dataset = dataset
            self.dataloader = TorchDataLoader(
                dataset = dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                num_workers = num_workers,
                collate_fn = custom_collate_fn
            )
            self._become_available()

    @Component.check_whether_available
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

    def check_whether_valid_size(self, size):
        return (type(size) is int) and (size > 0)

    def check_whether_valid_flag(self, flag):
        return (type(flag) is bool)

    def check_whether_valid_dataset(self, dataset):
        return isinstance(dataset, DatasetBase) and dataset.is_available

    @Component.check_whether_available
    def add(
        self,
        collection
    ):
        self.dataset.add(collection = collection)

    @Component.check_whether_available
    def save(self, collection):
        self.add(collection = collection)


class DataLoader(BaseDataLoader):

    def __init__(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    ):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )
