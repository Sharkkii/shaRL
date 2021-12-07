#### IO Helper ####

import os
import json
import pandas as pd

from .fs_helper import FileSystemHelper


class IOHelper(FileSystemHelper):

    def __init__(
        self,
        root,
        default_path = ""
    ):
        super().__init__(
            root = root,
            default_path = default_path
        )
        self.csv_root = self.abs_path(os.path.join(default_path, "csv"))
        self.json_root = self.abs_path(os.path.join(default_path, "json"))

    def setup(
        self
    ):
        super().setup()
        if (not os.path.isdir(self.csv_root)):
            os.makedirs(self.csv_root)
        if (not os.path.isdir(self.json_root)):
            os.makedirs(self.json_root)
    
    def reset(
        self
    ):
        super().reset()

    def abs_csv_path(
        self,
        path
    ):
        assert(len(path) > 0)
        ext = ".csv"
        path = os.path.abspath(os.path.join(self.csv_root, path)) + ext
        return path
    
    def abs_json_path(
        self,
        path
    ):
        assert(len(path) > 0)
        ext = ".json"
        path = os.path.abspath(os.path.join(self.json_root, path)) + ext
        return path

    def load_from_csv(
        self,
        path = "default"
    ):
        path = self.abs_csv_path(path)
        data = pd.read_csv(path).to_dict()
        return data

    def load_from_json(
        self,
        path = "default"
    ):
        path = self.abs_json_path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def save_to_csv(
        self,
        data,
        path = "default"
    ):
        assert(type(data) is dict)
        path = self.abs_csv_path(path)
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def save_to_json(
        self,
        data,
        path = "default"
    ):
        assert(type(data) is dict)
        path = self.abs_json_path(path)
        with open(path, "w") as f:
            json.dump(data, f, indent = 4)
