import argparse
import numpy as np
import pandas as pd
import tqdm
import time

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


def generate_one_signal(df, steps_num: int = 1024, sample_coef: float = 0.5):
    """
    Generates one output for given df
    :param df: df with info of given detector and event
    :param steps_num: number of timestamps by which time will be splitted
    :param sample_coef: percent of data to take for each step
    :return: np.array [steps_num] with energies
    """
    min_timestamp = min(df['timestamp'])
    max_timestamp = max(df['timestamp'])
    step = (max_timestamp - min_timestamp) / steps_num

    step_energies = []
    for i in range(steps_num):
        step_df = df[df['timestamp'] > i * step]
        step_df = step_df[step_df['timestamp'] < (i + 1) * step]
        step_df = step_df.sample(int(len(step_df) * sample_coef))  # randomly sample data
        step_energy = sum(step_df['energy'])
        step_energies.append(step_energy)

    return np.array(step_energies)


def get_detector_event_df(df_full, detector: int, event: int = -1):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    """
    Given full df returns df for given detector and event
    :param df_full:
    :param detector:
    :param event:
    :return: 
    """
    if event == -1:
        df = df_full[df_full['detector'] == detector]
    else:
        df = df_full[(df_full['detector'] == detector) & (df_full['event'] == event)]
    df.sort_values(by=['timestamp'], inplace=True)

    return df


def generate_signals(df, data_size: int,  use_postprocessing: bool, steps_num: int = 1024, sample_coef: float = 0.5):
    """
    Generates data for a given detector
    :param df_full: pandas df, output of get_events_df()
    :param data_size: number of samples to get
    :param use_postprocessing: whether to use output before or after photodetector
    :param steps_num: number of timestamps by which time will be splitted
    :param sample_coef: percent of data to take for each step
    :return: np.array with generated events
    """
    output_signals = []
    for _ in tqdm.tqdm(range(data_size)):
        output_signal = generate_one_signal(df, steps_num, sample_coef)
        if use_postprocessing:
            output_signal = postprocess_signal(output_signal)
        output_signals.append(output_signal)

    return np.array(output_signals)


def postprocess_signal(signal):
    """
    Getting result signal after photodetector
    :param signal: Output from generate_detector_event_output
    :return: processed signal
    """
    def build_kernel(x_cur, energy, x_min, x_max):
        kernel = lambda x: ((x - x_cur) ** 2) / np.exp((x - x_cur) / PROCESSING_TIME_NORM_COEF)
        x_linspace = np.linspace(x_min, x_max, x_max - x_min)
        y_linspace = energy * np.array(list(map(kernel, x_linspace)))
        y_linspace[:x_cur] = np.zeros(x_cur)
        return y_linspace

    result = np.zeros(len(signal))
    for x_cur, energy in enumerate(signal):
        y_cur = build_kernel(x_cur, energy, x_min=0, x_max=len(signal))
        result += y_cur
    return result


def main():
    args = parse_args()
    df = get_events_df(args.data_folder)
    print(df.head())


if __name__ == '__main__':
    main()
