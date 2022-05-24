#### DataLoader ####

from abc import ABCMeta, abstractmethod
from tabnanny import check
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data._utils.collate import default_collate

from ..common import check_whether_available
from .dataset import BaseDataset


def custom_collate_fn(batch):
    if (all([(item is None) for item in batch])):
        return None
    return default_collate(batch)


class BaseDataLoader(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True
    ):
        self.dataset = None
        self.dataloader = None
        self.batch_size = None
        self.shuffle = None
        self._is_available = False
        self.setup(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )
        
    def reset(
        self
    ):
        pass

    def setup(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True
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
                collate_fn = custom_collate_fn
            )
            self._become_available()

    @check_whether_available
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

    def check_whether_valid_size(self, size):
        return (type(size) is int) and (size > 0)

    def check_whether_valid_flag(self, flag):
        return (type(flag) is bool)

    def check_whether_valid_dataset(self, dataset):
        return isinstance(dataset, BaseDataset) and dataset.is_available


class DataLoader(BaseDataLoader):

    def __init__(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True
    ):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )

    def reset(
        self
    ):
        super().reset()

    def setup(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True
    ):
        super().setup(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )
