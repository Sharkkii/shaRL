#### Dataset ####

from ...common.data import SARS
from ...common.data import SGASG

from .base import MemoryBase
from .mixin import MemoryMixin
from .mixin import StepwiseMemoryMixin
from .mixin import EpisodewiseMemoryMixin


# T_STATE = torch.tensor
# T_ACTION = int
# T_REWARD = float


MAX_SIZE = 10000


class Dataset(MemoryMixin, MemoryBase):
    pass


class SarsDataset(StepwiseMemoryMixin, MemoryBase):

    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE
    ):
        StepwiseMemoryMixin.__init__(
            self,
            collection = collection,
            transform = transform,
            max_size = max_size
        )
        SarsDataset.setup(
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
        if (collection is None): return
        if (not self.check_whether_valid_sars_collection(collection)):
            del self._collection
            del self._transform
            del self._max_size
            raise ValueError("`collection` must be 'List[SARS]' object.")
    
    def getitem_wrapper(
        getitem
    ):
        def wrapper(
            self,
            index
        ):
            sars = getitem(self, index = index)
            return (sars.state, sars.action, sars.reward, sars.next_state)
        return wrapper

    @getitem_wrapper
    def __getitem__(
        self,
        index
    ):
        return StepwiseMemoryMixin.__getitem__(self, index = index)

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        if (not self.check_whether_valid_sars_collection(collection)):
            raise ValueError("`collection` must be 'List[SARS]' object.")

        StepwiseMemoryMixin.add_collection(self, collection = collection)

    def check_whether_valid_sars_collection(self, collection):
        return all([ (type(item) is SARS) for item in collection ])


class SgasgDataset(StepwiseMemoryMixin, MemoryBase):

    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_SIZE
    ):
        StepwiseMemoryMixin.__init__(
            self,
            collection = collection,
            transform = transform,
            max_size = max_size
        )
        SgasgDataset.setup(
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
        if (collection is None): return
        if (not self.check_whether_valid_sgasg_collection(collection)):
            del self._collection
            del self._transform
            del self._max_size
            raise ValueError("`collection` must be 'List[SGASG]' object.")
    
    def getitem_wrapper(
        getitem
    ):
        def wrapper(
            self,
            index
        ):
            sgasg = super().__getitem__(index = index)
            return (sgasg.state, sgasg.goal, sgasg.action, sgasg.next_state, sgasg.next_goal)
        return wrapper

    @getitem_wrapper
    def __getitem__(
        self,
        index
    ):
        return StepwiseMemoryMixin.__getitem__(self, index = index)

    def add_collection(
        self,
        collection
    ):
        if (not self.check_whether_valid_collection(collection)):
            raise ValueError("`collection` must be 'List' object.")
        if (not self.check_whether_valid_sgasg_collection(collection)):
            raise ValueError("`collection` must be 'List[SGASG]' object.")

        super().add_collection(collection)

    def check_whether_valid_sgasg_collection(self, collection):
        return all([ (type(item) is SGASG) for item in collection ])
