import time
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help='Data folder where files are stored')
    parser.add_argument('--to_save', action="store_false", help='Whether to save df with events')
    return parser.parse_args()


def process_file(data, attr_name: str, data_folder: str) -> None:
    """
    Adds values of corresponding attribute to data
    :param data: Dictionary where keys are attributes' names
    :param attr_name: attribute name
    :param data_folder: path where to read data
    :return:
    """
    with open("{}/{}.txt".format(data_folder, attr_name)) as file_:
        for line in file_:
            data[attr_name].append(float(line.strip()))


def get_events_df():
    """
    :return: events df, where each column is an attribute
    """
    attributes = ['x', 'y', 'z', 'energy', 'detector', 'timestamp', 'event']
    data = {attr_name: [] for attr_name in attributes}

    for attr_name in attributes:
        print("Attribute: {}".format(attr_name))
        time.sleep(1)
        process_file(data, attr_name)
        time.sleep(1)

    # Check that all attributes have
    # the same number of values
    attrs_values = data.values()
    data_size = len(attrs_values[0])
    assert all(len(item) == data_size for item in attrs_values)

    df = pd.DataFrame(data)
    return df


def get_detector_event_data(df_full, detector: int, event: int):
    df = df_full[df_full['detector'] == detector]
    df = df[df['event'] == event]
    df.sort_values(by='timestamp', inplace=True)

    return df


def generate_detector_event_output(df_full, detector: int, event: int, steps_num: int = 500):
    """
    Generates one output for given detector and event
    :param df_full: df with info of given detector and event
    :param detector: detector number
    :param event: event number
    :param steps_num: number of timestamps by which time will be splitted
    :return: np.array [steps_num] with energies
    """
    df = get_detector_event_data(df_full, detector, event)
    min_timestamp = min(df['timestamp'])
    max_timestamp = max(df['timestamp'])
    step = (max_timestamp - min_timestamp) / steps_num

    step_energies = []
    for i in range(steps_num):
        step_df = df[df['timestamp'] > i * step]
        step_df = step_df[step_df['timestamp'] < (i + 1) * step]
        step_df = step_df.sample(len(step_df) // 2)  # randomly sample half of data
        step_energy = sum(step_df['energy'])
        step_energies.append(step_energy)

    return np.array(step_energies)


def main():
    df = get_events_df()
    print(df.head())


if __name__ == '__main__':
    main()
