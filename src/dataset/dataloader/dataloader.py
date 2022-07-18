#### DataLoader ####

from .base import DataLoaderBase
from .mixin import DataLoaderMixin


class DataLoader(DataLoaderMixin, DataLoaderBase):

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
