#### Record Helper ####

from .io_helper import IOHelper

class RecordHelper(IOHelper):

    def __init__(
        self,
        root,
        n_capacity = 1 # will be ignored
    ):
        super().__init__(
            root = root
        )
        self.n_capacity = n_capacity
        self.cell = {}
    
    def reset(
        self
    ):
        self.cell = {}
    
    def setup(
        self
    ):
        super().setup()

    def record(
        self,
        key,
        value
    ):
        if (key in self.cell):
            self.cell[key].append(value)
        else:
            self.cell[key] = [ value ]

    def save_csv(
        self,
        path
    ):
        super().save_csv(self.cell, path)
    
    def save_json(
        self,
        path
    ):
        super().save_json(self.cell, path)

