import time
import numpy as np
import pandas as pd
import argparse
from generation.config import PROCESSING_TIME_NORM_COEF


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


def get_events_df(data_folder):
    """
    :param data_folder:
    :return: events df, where each column is an attribute
    """
    attributes = ['x', 'y', 'z', 'energy', 'detector', 'timestamp', 'event']
    data = {attr_name: [] for attr_name in attributes}

    for attr_name in attributes:
        print("Attribute: {}".format(attr_name))
        time.sleep(1)
        process_file(data, attr_name, data_folder)
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


def generate_detector_event_output(df_full, detector: int, event: int, steps_num: int = 1024):
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


def postprocess_output(step_energies):
    """
    Getting result signal after photodetector
    :param step_energies: Output from generate_detector_event_output
    :return: processed signal
    """
    def build_kernel(x_cur, energy, x_min, x_max):
        kernel = lambda x: ((x - x_cur) ** 2) / np.exp((x - x_cur) / PROCESSING_TIME_NORM_COEF)
        x_linspace = np.linspace(x_min, x_max, x_max - x_min)
        y_linspace = energy * np.array(list(map(kernel, x_linspace)))
        y_linspace[:x_cur] = np.zeros(x_cur)
        return y_linspace

    result = np.zeros(len(step_energies))
    for x_cur, energy in enumerate(step_energies):
        y_cur = build_kernel(x_cur, energy, x_min=0, x_max=len(step_energies))
        result += y_cur
    return result


def main():
    args = parse_args()
    df = get_events_df(args.data_folder)
    print(df.head())


if __name__ == '__main__':
    main()
