#### Data Augmentator ####

from abc import ABC, abstractmethod


class DataAugmentatorBase(ABC):

    @abstractmethod
    def __init__(self): raise NotImplementedError
    @abstractmethod
    def augment(self): raise NotImplementedError

    def add_decorator(
        self,
        add_fn
    ):
        def wrapper(dataset, collection):
            augmentation = self.augment(collection)
            if (augmentation):
                collection = collection + augmentation
            add_fn(dataset, collection)
        return wrapper


class DataAugmentator(DataAugmentatorBase):

    def __init__(
        self
    ):
        pass

    def augment(
        self,
        data
    ):
        return None


class CustomDataAugmentator(DataAugmentatorBase):

    def __init__(
        self,
        augment_fn
    ):
        self._fn = augment_fn
    
    def augment(
        self,
        data
    ):
        return self._fn(data)
