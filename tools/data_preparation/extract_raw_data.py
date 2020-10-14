import os

import numpy as np
import pandas as pd
import uproot

from generation.config import ROOT_FILE_PATH, ROOT_TREE_NAME, ATTRIBUTES, SPACAL_DATA_PATH, INT_ATTRIBUTES


def _prepare_attributes_df(root_tree, attrs=ATTRIBUTES, res_path=SPACAL_DATA_PATH) -> None:  # TODO: (@whiteRa2bit, 2020-08-25) Add types
    """
    Creates df containing attributes and saves it at given path
    :param data: Dictionary where keys are attributes' names
    :param attr_name: attribute name
    :param data_path: path to attribute file
    :return:
    """

    def _process_line(line, line_type):
        return line_type(line.strip())

    if os.path.exists(res_path):
        print(f"Attributes dataframe at path {res_path} already exsists")  # TODO: (@whiteRa2bit, 2020-08-30) Add logger
        return

    data = {attr: [] for attr in attrs}
    for attr in attrs:
        print(f'Processing attribute: {attr}')
        attr_values = root_tree.array(attr)
        attr_type = "int16" if attr in INT_ATTRIBUTES else "float32"
        data[attr] = attr_values.astype(attr_type)

    # Check that all attributes have the same number of values
    attr_values = list(data.values())
    assert all(len(item) == len(attr_values[0]) for item in attr_values)

    df = pd.DataFrame(data)
    df.to_parquet(res_path)


def main():
    root_file = uproot.open(ROOT_FILE_PATH)
    root_tree = root_file[ROOT_TREE_NAME]
    _prepare_attributes_df(root_tree)


if __name__ == '__main__':
    main()
