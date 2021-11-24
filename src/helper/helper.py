#### Helper ####

import os

class BaseHelper:

    def __init__(
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
        self.setup_directory()

    def setup_directory(
        self
    ):
        # /dest/
        if (not os.path.isdir(self.dest)):
            os.mkdir(self.dest)
