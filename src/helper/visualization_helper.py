#### Visualization Helper ####

import os
import subprocess
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from .io_helper import IOHelper

class VisualizationHelper(IOHelper):

    def __init__(
        self,
        root,
        default_path = ""
    ):
        super().__init__(
            root = root,
            default_path = default_path
        )
        self.root = root
        self.png_root = self.abs_path(os.path.join(default_path, "png"))
        self.figure = None
        self.axes = None

    def setup(
        self
    ):
        super().setup()
        self.setup_directory()
        if (self.figure is None):
            self.figure = plt.figure(0, figsize=(8,6))
        if (self.axes is None):
            self.axes = self.figure.add_subplot(111)

    def setup_directory(
        self
    ):
        if (not os.path.isdir(self.png_root)):
            os.makedirs(self.png_root)

    def abs_png_path(
        self,
        path
    ):
        assert(len(path) > 0)
        ext = ".png"
        path = os.path.abspath(os.path.join(self.png_root, path)) + ext
        return path

    def line(
        self,
        data # dict<key,value>
    ):
        assert(type(data) is dict)
        # FIXME:
        assert(len(data) == 1)

        (key, value), = data.items()
        x = range(len(value))
        y = value

        self.axes.clear()
        self.axes.plot(x, y, label=key)
        self.axes.legend(loc = "best")
        return self.figure

    def save_graph(
        self,
        graph, # figure
        path = "default"
    ):
        path = self.abs_png_path(path)
        graph.savefig(path)

    def load_graph(
        self,
        path = "default"
    ):
        path = self.abs_png_path(path)
        graph = plt.imread(path)
        return graph
    
    def show_graph(
        self,
        path = "default"
    ):
        path = self.abs_png_path(path)
        print(f"VisualizationHelper.show_graph: open {path}")
        command = ["open", path]
        subprocess.run(command)


# timestamp
def date2ymdhms(
    date # datetime.datetime.now()
):
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    hour = str(date.hour)
    minute = str(date.minute)
    second = str(date.second)
    date_string = year + "-" + month + "-" + day + "-" + hour + "-" + minute + "-" + second
    return date_string