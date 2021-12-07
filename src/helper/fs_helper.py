#### File System Helper ####

import os

from .helper import BaseHelper


class FileSystemHelper(BaseHelper):

    def __init__(
        self,
        root,
        default_path = "" # relative path
    ):
        super().__init__()
        self._root = root
        self._default_path = default_path

    @property
    def root(self):
        return self._root
    
    @root.setter
    def root(self, path):
        pass

    @property
    def default_path(self):
        return self._default_path
    
    @default_path.setter
    def default_path(self, path):
        pass
    
    def setup(
        self
    ):
        super().setup()
        if (not os.path.isdir(self.root)):
            os.makedirs(self.root)

    def reset(
        self
    ):
        super().reset()

    def abs_path(
        self,
        path, # relative path
    ):
        path = os.path.abspath(os.path.join(self.root, path))
        return path
    
    def abs_path_or_default(
        self,
        path = ""
    ):
        if (len(path) > 0):
            return self.abs_path(path)
        elif (len(self.default_path) > 0):
            return self.default_path
        else:
            return ""
