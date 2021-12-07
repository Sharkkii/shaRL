import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from src.helper import RecordHelper
from src.helper import VisualizationHelper

def test_add_record_helper():

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dest'))
    helper = RecordHelper(
        root = root
    )
    helper.setup()

    for _ in range(10):
        r = np.random.randn()
        helper.record("random variable A", r)
        helper.record("random variable B", r)
    helper.save_to_csv("result")
    helper.save_to_json("result")
    data = helper.load_from_csv("result")
    print(data)
    data = helper.load_from_json("result")
    print(data)

    print("OK: test_add_record_helper")

def test_add_visualization_helper():

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dest"))
    vhelper = VisualizationHelper(
        root = root
    )
    vhelper.setup()

    data = {"data": [3, 1, 4, 1, 5]}
    print(data)
    graph = vhelper.line(data)
    vhelper.save_graph(
        graph,
        path = "result"
    )
    image = vhelper.load_graph("result")
    vhelper.show_graph("result")

    print("OK: test_add_visualization_helper")

def test_add_helper():
    test_add_record_helper()
    test_add_visualization_helper()

def main():
    test_add_helper()

if __name__ == "__main__":
    main()