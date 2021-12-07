#### Helper ####

import os

class BaseHelper:

    def __init__(
        self
    ):
        pass

    def setup(
        self
    ):
        pass

    def reset(
        self
    ):
        pass

class Helper(BaseHelper):

    def __init__(
        self,
        root
    ):
        self.root = root
        self.dest = os.path.join(root, "dest")
    
    def setup(
        self
    ):
        pass

    def reset(
        self
    ):
        pass
