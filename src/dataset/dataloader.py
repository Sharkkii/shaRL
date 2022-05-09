#### DataLoader ####

from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import BaseDataset


class BaseDataLoader(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        dataset = None,
        batch_size = 1,
        shuffle = True
    ):
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
            self.shuffle = True

        if (
            self.check_whether_valid_size(self.batch_size) and self.check_whether_valid_flag(self.shuffle) and
            self.check_whether_valid_dataset(dataset)
        ):
            self.dataloader = TorchDataLoader(
                dataset = dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )
            self._become_available()

    def __iter__(
        self
    ):
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
        return isinstance(dataset, BaseDataset)

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
