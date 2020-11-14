import os

import numpy as np
import pandas as pd
import uproot
from loguru import logger

from generation.utils import timer
from generation.config import ROOT_FILES_DIR, ROOT_TREE_NAME, EVENT_ATTR, ATTRIBUTES, SPACAL_DATA_PATH, INT_ATTRIBUTES


@timer
def _prepare_attributes_df(root_trees, attrs=ATTRIBUTES,
                           res_path=SPACAL_DATA_PATH) -> None:  # TODO: (@whiteRa2bit, 2020-08-25) Add types
    """
    Creates df containing attributes and saves it at given path
    :param root_trees: Root trees
    :param attr_name: attribute name
    :param res_path: path to attribute file
    :return:
    """

    def _process_line(line, line_type):
        return line_type(line.strip())

    def _numerate_events(events_lists):
        """
        Function to renumerate events_lists
        :param events_lists: List of lists with event numbers
        :return: List of lists with new event numbers, so that
        they do not repeat
        """
        max_nums = [max(events) for events in events_lists]
        start_nums = np.cumsum([num + 1 for num in max_nums])
        for i in range(1, len(events_lists)):
            events_lists[i] += start_nums[i - 1]

        return events_lists

    if os.path.exists(res_path):
        logger.info(f"Attributes dataframe at path {res_path} already exsists")
        return

    data = {attr: [] for attr in attrs}
    for attr in attrs:
        logger.info(f'Processing attribute: {attr}')
        attr_values = [tree.array(attr) for tree in root_trees]
        attr_values = _numerate_events(attr_values) if attr == EVENT_ATTR else attr_values
        attr_values = np.concatenate(attr_values)
        attr_type = "int16" if attr in INT_ATTRIBUTES else "float32"
        data[attr] = attr_values.astype(attr_type)

    # Check that all attributes have the same number of values
    attr_values = list(data.values())
    assert all(len(item) == len(attr_values[0]) for item in attr_values)

    df = pd.DataFrame(data)
    # Filter only back layer
    df = df[df["z"] < 0]  # TODO: (@whiteRa2bit, 2020-11-14) Fix OOM
    df.to_parquet(res_path)


def main():
    filenames = os.listdir(ROOT_FILES_DIR)
    logger.info(f"{len(filenames)} files were found: {filenames}")
    filepaths = [os.path.join(ROOT_FILES_DIR, name) for name in filenames]
    root_files = [uproot.open(path) for path in filepaths]
    root_trees = [root_file[ROOT_TREE_NAME] for root_file in root_files]
    _prepare_attributes_df(root_trees)


if __name__ == '__main__':
    main()
