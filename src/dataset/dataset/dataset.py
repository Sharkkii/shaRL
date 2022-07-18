#### Dataset ####

import torch
from ...common.data import SARS
from ...common.data import SGASG

from .base import DatasetBase
from .base import MemoryBase
from .mixin import DatasetMixin
from .mixin import MemoryMixin
from .mixin import StepwiseMemoryMixin
from .mixin import EpisodeMemoryMixin

# T_STATE = torch.tensor
# T_ACTION = int
# T_REWARD = float


MAX_SIZE = 10000
MAX_STEP_SIZE = 10000
MAX_EPISODE_SIZE = 100


class StepwiseSARSMemory(StepwiseMemoryMixin, MemoryBase):

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
        StepwiseSARSMemory.setup(
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


SarsDataset = StepwiseSARSMemory


class StepwiseSGASGMemory(StepwiseMemoryMixin, MemoryBase):

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
        StepwiseSGASGMemory.setup(
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
            sgasg = getitem(self, index = index)
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


SgasgDataset = StepwiseSGASGMemory


class SARSEpisodeMemory(EpisodeMemoryMixin, MemoryBase):

    def __init__(
        self,
        collection = None,
        transform = None,
        max_size = MAX_EPISODE_SIZE
    ):
        EpisodeMemoryMixin.__init__(
            self,
            collection = collection,
            transform = transform,
            max_size = max_size
        )
        SARSEpisodeMemory.setup(
            self,
            collection = collection,
            transform = transform,
            max_size = max_size   
        )

    def setup(
        self,
        collection = None,
        transform = None,
        max_size = MAX_EPISODE_SIZE
    ):
        if (collection is None): return
        if (not self.check_whether_valid_sars_episode_collection):
            del self._collection
            del self._transform
            del self._max_size
            raise ValueError("`collection` must be 'List[List[SARS]]`.")
    
    def getitem_wrapper(
        getitem
    ):
        def wrapper(
            self,
            index
        ):
            sars_episode = getitem(self, index = index)
            state_episode = torch.stack([ sars.state for sars in sars_episode ])
            action_episode = torch.stack([ sars.action for sars in sars_episode ])
            reward_episode = torch.stack([ sars.reward for sars in sars_episode ])
            next_state_episode = torch.stack([ sars.next_state for sars in sars_episode ])
            return (state_episode, action_episode, reward_episode, next_state_episode)
        return wrapper

    @getitem_wrapper
    def __getitem__(
        self,
        index
    ):
        return EpisodeMemoryMixin.__getitem__(self, index = index)

    def check_whether_valid_episode_collection(self, collection):
        return all([ (type(episode) is list) for episode in collection ])

    def check_whether_valid_sars_episode_collection(self, collection):
        return all([ all([(type(step) is SARS) for step in episode]) for episode in collection ])


class Dataset(DatasetMixin, DatasetBase):
    pass


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
