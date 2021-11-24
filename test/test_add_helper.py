import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.helper import Helper, VisualizationHelper

def test_add_helper():

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    helper = Helper(
        root = root
    )
    dest = helper.dest
    project_root = os.path.join(dest, "project")
    print(project_root)
    vhelper = VisualizationHelper(
        root = project_root
    )
    vhelper.setup()

    data = {"data": [3, 1, 4, 1, 5]}
    print(data)
    graph = vhelper.line(data)
    vhelper.save_csv_data(data)
    vhelper.save_json_data(data)
    vhelper.save_graph(graph)

    data = vhelper.load_csv_data()
    print(data)
    data = vhelper.load_json_data()
    print(data)
    image = vhelper.load_graph()
    vhelper.show_graph(image)

    print("OK: test_add_helper")

def main():
    test_add_helper()

if __name__ == "__main__":
    main()