#### Visualization Helper ####

import os
import subprocess
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from .helper import BaseHelper

class VisualizationHelper(BaseHelper):

    def __init__(
        self,
        root
    ):
        super().__init__()
        # /dest/*/
        self.root = root
        # /dest/*/data/
        self.data = os.path.join(self.root, "data")
        # /dest/*/images/
        self.images = os.path.join(self.root, "images")
        # /dest/*/data/csv/
        self.csv_dir_path = os.path.join(self.data, "csv")
        # /dest/*/data/json/
        self.json_dir_path = os.path.join(self.data, "json")
        # /dest/*/images/png/
        self.png_dir_path = os.path.join(self.images, "png")
        self.csv_file_path = ""
        self.json_file_path = ""
        self.png_file_path = ""
        self.figure = None
        self.axes = None

    def setup(
        self
    ):
        self.setup_directory()
        if (self.figure is None):
            self.figure = plt.figure(0, figsize=(8,6))
        if (self.axes is None):
            self.axes = self.figure.add_subplot(111)

    def setup_directory(
        self
    ):
        # /dest/*/
        if (not os.path.isdir(self.root)):
            os.makedirs(self.root)

        # /dest/*/data/
        if (not os.path.isdir(self.data)):
            os.mkdir(self.data)

        # /dest/*/images/
        if (not os.path.isdir(self.images)):
            os.mkdir(self.images)

        # /dest/*/data/csv/
        if (not os.path.isdir(self.csv_dir_path)):
            os.mkdir(self.csv_dir_path)

        # /dest/*/data/json/
        if (not os.path.isdir(self.json_dir_path)):
            os.mkdir(self.json_dir_path)

        # /images/png/
        if (not os.path.isdir(self.png_dir_path)):
            os.mkdir(self.png_dir_path)

    def save_csv_data(
        self,
        data # dict<key,value>
    ):
        assert(type(data) is dict)
        self.setup()
        timestamp = date2ymdhms(datetime.now())
        self.csv_file_path = path = os.path.join(self.csv_dir_path, timestamp + ".csv")
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def load_csv_data(
        self
    ):
        path = self.csv_file_path
        data = pd.read_csv(path).to_dict()
        return data

    def save_json_data(
        self,
        data # dict<key,value>
    ):
        timestamp = date2ymdhms(datetime.now())
        self.json_file_path = path = os.path.join(self.json_dir_path, timestamp + ".json")
        with open(path, "w") as f:
            json.dump(data, f, indent = 4)

    def load_json_data(
        self
    ):
        path = self.json_file_path
        with open(path, "r") as f:
            data = json.load(f)
        return data

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
        self.axes.legend(loc="best")
        return self.figure

    def save_graph(
        self,
        graph # figure
    ):
        self.setup()
        timestamp = date2ymdhms(datetime.now())
        self.png_file_path = path = os.path.join(self.png_dir_path, timestamp + ".png")
        graph.savefig(path)

    def load_graph(
        self
    ):
        path = self.png_file_path
        graph = plt.imread(path)
        return graph
    
    def show_graph(
        self,
        graph
    ):
        command = ["open", self.png_file_path]
        subprocess.run(command)
        # FIXME: `imshow` doesn't work well...
        # self.axes.clear()
        # self.axes.imshow(graph)
        # plt.show()


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