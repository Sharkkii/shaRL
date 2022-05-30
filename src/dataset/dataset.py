#### Dataset ####

from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset as TorchDataset

from ..common import Component
from ..common.data import SARS
from ..common.data import SAGS


# T_STATE = torch.tensor
# T_ACTION = int
# T_REWARD = float


MAX_SIZE = 10000

class BaseDataset(Component, TorchDataset, metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE,
    ):
        Component.__init__(self)
        self.collection = None      
        self.transform = None
        self.max_size = None
        self.setup(
            collection = collection,
            transform = transform,
            max_size = max_size
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
        transform = None,
        max_size = MAX_SIZE
    ):
        if (collection is None):
            return

        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        
        self.collection = collection
        self.max_size = max_size
        if (self.check_whether_valid_transform(transform)):
            self.transform = transform
        self.trancate()

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

    @property
    def size(self):
        return len(self.collection)

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

    def trancate(
        self
    ):
        del self.collection[:-self.max_size]

    def check_whether_valid_collection(self, collection):
        return (collection is not None) and (hasattr(collection, "__iter__")) and (hasattr(collection, "__getitem__"))

    def check_whether_valid_transform(self, transform):
        return callable(transform)


class Dataset(BaseDataset):

    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE
    ):
        super().__init__(
            collection = collection,
            transform = transform,
            max_size = max_size
        )

    def reset(
        self
    ):
        super().reset()

    def setup(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE
    ):
        super().setup(
            collection = collection,
            transform = transform,
            max_size = max_size
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
        transform = None,
        max_size = MAX_SIZE
    ):
        super().__init__(
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

        if (not self.check_whether_valid_sars_collection(collection)):
            raise ValueError("`collection` must be 'List[SARS]' object.")

        super().setup(
            collection = collection,
            transform = transform,
            max_size = max_size
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


class SagsDataset(Dataset):

    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE
    ):
        super().__init__(
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

        if (not self.check_whether_valid_sag_collection(collection)):
            raise ValueError("`collection` must be 'List[SAGS]' object.")

        super().setup(
            collection = collection,
            transform = transform,
            max_size = max_size
        )

    def __getitem__(
        self,
        index
    ):
        sags = super().__getitem__(index = index)
        return (sags.state, sags.action, sags.goal, sags.next_state)

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")

        if (not self.check_whether_valid_sag_collection(collection)):
            raise ValueError("`collection` must be 'List[SAGS]' object.")

        super().add_collection(collection)

    def check_whether_valid_sag_collection(self, collection):
        return all([ (type(item) is SAGS) for item in collection ])
