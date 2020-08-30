import functools
import time
import os

import numpy as np
import pandas as pd
import tqdm

from generation.dataset.dataset import get_attributes_df, _get_event_dir, get_detector_event_df, \
    get_detector_event_df_path, get_detector_event_signal_path, get_detector_event_signal, generate_one_signal
from generation.config import DF_DIR, SIGNAL_DIR, PROCESSED_SIGNAL_DIR, ATTRIBUTES, ATTRIBUTE_PATHS, \
    INT_ATTRIBUTES, SPACAL_DATA_PATH, STEPS_NUM


def _prepare_attributes_df(attrs=ATTRIBUTES, attr_paths=ATTRIBUTE_PATHS, res_path=SPACAL_DATA_PATH) -> None:  # TODO: (@whiteRa2bit, 2020-08-25) Add types
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
    for attr, attr_path in zip(attrs, attr_paths):
        print(f'Processing attribute: {attr}')
        attr_values = np.load(attr_path)
        attr_type = int if attr in INT_ATTRIBUTES else float
        data[attr] = attr_values.astype(attr_type)

    # Check that all attributes have the same number of values
    attr_values = list(data.values())
    assert all(len(item) == len(attr_values[0]) for item in attr_values)

    df = pd.DataFrame(data)
    df.to_pickle(res_path)


def _create_dirs(df_full, df_dir: str = DF_DIR, signal_dir: str = SIGNAL_DIR):
    def create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)
        
    events = df_full['event'].unique()
    create_dir(df_dir)
    create_dir(signal_dir)
    for event in events:
        create_dir(_get_event_dir(df_dir, event))
        create_dir(_get_event_dir(signal_dir, event))


def _prepare_event_df(df_full, event: int, df_dir: str = DF_DIR):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    """
    Given full df returns df for given detector and event
    :param df_full:
    :param detector:
    :param event:
    :return: 
    """
    detectors = df_full['detector'].unique()  # TODO: (@whiteRa2bit, 2020-08-25) Fix call by attribute name    
    event_df = df_full[df_full['event'] == event]  # TODO: (@whiteRa2bit, 2020-08-25) Fix call by attribute name
    for detector in detectors:
        df_path = get_detector_event_df_path(detector, event)
        detector_event_df = event_df[event_df['detector'] == detector]  # TODO: (@whiteRa2bit, 2020-08-25) Fix call by attribute name
        detector_event_df.to_csv(df_path, index=False)

    return event_df


def _prepare_detector_event_signal(detector: int, event: int, signal_dir: str = SIGNAL_DIR):
    df = get_detector_event_df(detector, event)
    signal_path = get_detector_event_signal_path(detector, event)
    signal = generate_one_signal(df)
    np.save(signal_path, signal)


def main(): 
    _prepare_attributes_df()
    df = get_attributes_df()
    _create_dirs(df)
    events = df['event'].unique()
    detectors = df['detector'].unique()

    for event in events:
        print(f'Preparing {event} event df...')
        event_df = _prepare_event_df(df, event)
        print(f'Preparing detectors signals for event {event}')
        for detector in tqdm.tqdm(detectors):
            _prepare_detector_event_signal(detector, event)


if __name__ == '__main__':
    main()
