#### Dataset ####

from ...common import Component
from ...common.data import SARS
from ...common.data import SGASG

from .mixin import DatasetMixin

# T_STATE = torch.tensor
# T_ACTION = int
# T_REWARD = float


MAX_SIZE = 10000


class Dataset(DatasetMixin):

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


class SgasgDataset(Dataset):

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
            raise ValueError("`collection` must be 'List[SGASG]' object.")

        super().setup(
            collection = collection,
            transform = transform,
            max_size = max_size
        )

    def __getitem__(
        self,
        index
    ):
        sgasg = super().__getitem__(index = index)
        return (sgasg.state, sgasg.goal, sgasg.action, sgasg.next_state, sgasg.next_goal)

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")

        if (not self.check_whether_valid_sag_collection(collection)):
            raise ValueError("`collection` must be 'List[SGASG]' object.")

        super().add_collection(collection)

    def check_whether_valid_sag_collection(self, collection):
        return all([ (type(item) is SGASG) for item in collection ])


class AugmentedDataset(Dataset):

    def __init__(
        self,
        collection = None,
        transform = None,
        data_augmentator = None,
        max_size = MAX_SIZE,
    ):
        Dataset.__init__(
            self,
            collection = collection,
            transform = transform,
            max_size = max_size
        )
        AugmentedDataset.setup(
            self,
            data_augmentator = data_augmentator
        )

    def setup(
        self,
        data_augmentator = None
    ):
        self.data_augmentator = data_augmentator

    def add_collection(self, collection):
        add_collection_fn = self.data_augmentator.add_decorator(Dataset.add_collection)
        return add_collection_fn(self, collection = collection)
