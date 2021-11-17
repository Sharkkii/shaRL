#### Helper ####

import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

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
        self.csv = os.path.join(self.data, "csv")
        # /dest/*/data/json/
        self.json = os.path.join(self.data, "json")
        # /dest/*/images/png/
        self.png = os.path.join(self.images, "png")

    def setup(
        self
    ):
        self.setup_directory()

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
        if (not os.path.isdir(self.csv)):
            os.mkdir(self.csv)

        # /dest/*/data/json/
        if (not os.path.isdir(self.json)):
            os.mkdir(self.json)

        # /images/png/
        if (not os.path.isdir(self.png)):
            os.mkdir(self.png)

    def save_graph(
        self,
        graph # figure
    ):
        self.setup()
        timestamp = date2ymdhms(datetime.now())
        path = os.path.join(self.png, timestamp + ".png")
        graph.savefig(path)

    def line(
        self,
        data # dict<key,value>
    ):
        assert(type(data) is dict)
        # FIXME:
        assert(len(data) == 1)

        figure = plt.figure(figsize=(8,6))
        axes = figure.add_subplot(1,1,1)

        (key, value), = data.items()
        x = range(len(value))
        y = value
        axes.plot(x, y, label=key)
        axes.legend(loc="best")
        return figure

    def save_csv_data(
        self,
        data # dict<key,value>
    ):
        assert(type(data) is dict)
        self.setup()
        timestamp = date2ymdhms(datetime.now())
        path = os.path.join(self.csv, timestamp + ".csv")
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def save_json_data(
        self,
        data # dict<key,value>
    ):
        timestamp = date2ymdhms(datetime.now())
        path = os.path.join(self.json, timestamp + ".json")
        with open(path, "w") as f:
            json.dump(data, f, indent = 4)

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