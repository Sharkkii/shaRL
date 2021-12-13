#### RL DataLoader ####

from torch.utils.data import DataLoader


class RLDataLoader:

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle = True
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )

    @property
    def is_available(
        self
    ):
        return self.dataset.is_available

    def __iter__(
        self
    ):
        return iter(self.dataloader)
    
    def load(
        self
    ):
        history = self.dataset.load()
        return history

    def save(
        self,
        history
    ):
        self.dataset.save(history)
