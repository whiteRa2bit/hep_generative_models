import time
import os

import numpy as np
import pandas as pd
import tqdm

from generation.dataset.dataset import get_attributes_df, _get_event_dir, get_detector_event_df, get_detector_event_df_path, \
                        get_detector_event_signal_path, get_detector_event_signal
from generation.config import DF_DIR, SIGNAL_DIR, PROCESSED_SIGNAL_DIR, \
                        ATTRIBUTES, ATTRIBUTE_PATHS, SPACAL_DATA_PATH, STEPS_NUM


def _prepare_attributes_df(attrs=ATTRIBUTES, attr_paths=ATTRIBUTE_PATHS, res_path=SPACAL_DATA_PATH) -> None:  # TODO: (@whiteRa2bit, 2020-08-25) Add types
    """
    Creates df containing attributes and saves it at given path
    :param data: Dictionary where keys are attributes' names
    :param attr_name: attribute name
    :param data_path: path to attribute file
    :return:
    """
    def process_line(line):
        return float(line.strip())

    data = {attr: [] for attr in attrs}
    for attr, attr_path in zip(attrs, attr_paths):
        print(f'Processing attribute: {attr}')
        with open(attr_path, 'r') as attr_file:
            data[attr] = list(map(process_line, attr_file.readlines()))

    # Check that all attributes have the same number of values
    attr_values = list(data.values())
    assert all(len(item) == len(attr_values[0]) for item in attr_values)

    df = pd.DataFrame(data)
    df.to_pickle(res_path)  # TODO: (@whiteRa2bit, 2020-08-25) Remove pickle


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
    event_df = df_full[df_full['event'] == event]  # TODO: (@whiteRa2bit, 2020-08-25) Fix call by attribute name
    detectors = event_df['detector'].unique()  # TODO: (@whiteRa2bit, 2020-08-25) Fix call by attribute name    
    for detector in detectors:
        df_path = get_detector_event_df_path(detector, event)
        detector_event_df = event_df[event_df['detector'] == detector]  # TODO: (@whiteRa2bit, 2020-08-25) Fix call by attribute name
        detector_event_df.to_csv(df_path, index=False)

    return event_df


def _generate_one_signal(df, steps_num: int = STEPS_NUM):
    """
    Generates one output for given df
    :param df: df with info of given detector and event
    :param steps_num: number of timestamps by which time will be splitted
    :return: np.array [steps_num] with energies
    """
    if df.empty:
        return np.zeros(steps_num)

    min_timestamp = min(df['timestamp'])
    max_timestamp = max(df['timestamp'])
    step = (max_timestamp - min_timestamp) / steps_num

    step_energies = []
    for i in range(steps_num):
        step_df = df[df['timestamp'] > i * step]
        step_df = step_df[step_df['timestamp'] < (i + 1) * step]
        step_energy = sum(step_df['energy'])
        step_energies.append(step_energy)

    return np.array(step_energies)


def _prepare_detector_event_signal(detector: int, event: int, signal_dir: str = SIGNAL_DIR):
    df = get_detector_event_df(detector, event)
    signal_path = get_detector_event_signal_path(detector, event)
    signal = _generate_one_signal(df)
    np.save(signal_path, signal)


def main(): 
    # df = _prepare_attributes_df()  # TODO: (@whiteRa2bit, 2020-08-25) Add call _prepare_attributes_df + check for existing data
    df = get_attributes_df()
    _create_dirs(df)
    events = df['event'].unique()

    for event in events:
        print(f'Preparing {event} event df...')
        event_df = _prepare_event_df(df, event)
        detectors = event_df['detector'].unique()
        print(f'Preparing detectors signals for event {event}')
        for detector in tqdm.tqdm(detectors):
            _prepare_detector_event_signal(detector, event)


if __name__ == '__main__':
    main()
