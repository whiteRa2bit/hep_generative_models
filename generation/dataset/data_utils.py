import argparse
import time
import os

import numpy as np
import pandas as pd
import tqdm
import h5py

from generation.config import DF_DIR, EVENTS_PATH, DETECTORS_PATH, FULL_SIGNALS_DIR, PROCESSING_TIME_NORM_COEF, \
    SIGNAL_DIM, SPACAL_DATA_PATH, FRAC_SIGNALS_DIR, H5_DATASET_NAME, POSTPROCESSED_SIGNALS_DIR


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_h5(data, dataset_name, path):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(dataset_name, data=data, compression='gzip')
    h5f.close()


def load_h5(path, dataset_name):
    h5f = h5py.File(path, 'r')
    dataset_h5 = h5f[dataset_name][:]
    h5f.close()
    return dataset_h5


def get_events(path=EVENTS_PATH):
    """
    Given path to events .npy file returns events
    param: path : path to events file
    return: events np array
    """
    events = np.load(path)
    return events


def get_detectors(path=DETECTORS_PATH):
    """
    Given path to detectors .npy file returns detectors
    param: path : path to detectors file
    return: detectors np array
    """
    detectors = np.load(path)
    return detectors


def get_event_dir(base_dir: str, event: int):
    """
    Given event and base directory return event directory
    :param base_dir: base directory
    :param event: event number
    :return: directory with files associated with given event
    """
    return os.path.join(base_dir, 'event_{}').format(event)


def get_event_detector_df_path(event: int, detector: int, df_dir: str = DF_DIR):
    """
    Given detector and event returns path to corresponding df
    :param event: event number
    :param detector: detector number
    :param df_dir: directory with dataframe files
    :return: path to pandas dataframe
    """
    event_dir = get_event_dir(df_dir, event)
    df_path = os.path.join(event_dir, 'detector_{}.parquet').format(detector)
    return df_path


def get_event_detector_df(event: int, detector: int):
    """
    Given detector and event returns corresponding df
    :param event: event number
    :param detector: detector number
    :return: pandas dataframe
    """
    df_path = get_event_detector_df_path(event, detector)
    df = pd.read_parquet(df_path)
    return df


def get_detector_signals_path(detector: int, signals_dir: str = FULL_SIGNALS_DIR):
    """
    Given detector and event returns path to signal
    :param detector: detector number
    :param signal_dir: directory with signal files
    :return: path to np array
    """
    signals_path = os.path.join(signals_dir, 'detector_{}.h5').format(detector)
    return signals_path


def get_detector_signals(detector: int):
    """
    Given detector and event returns corresponding signal
    :param event: event number
    :param detector: detector number
    :return: numpy array with shape SIGNAL_DIM
    """
    signals_path = get_detector_signals_path(detector)
    signals = load_h5(signals_path, H5_DATASET_NAME)
    return signals


def get_detector_postprocessed_signals_path(detector: int, signals_dir: str = POSTPROCESSED_SIGNALS_DIR):
    """
    Given detector and event returns path to signal
    :param detector: detector number
    :param signal_dir: directory with signal files
    :return: path to np array
    """
    signals_path = os.path.join(signals_dir, 'detector_{}.h5').format(detector)
    return signals_path


def get_detector_postprocessed_signals(detector: int):
    """
    Given detector and event returns corresponding signal
    :param event: event number
    :param detector: detector number
    :return: numpy array with shape SIGNAL_DIM
    """
    signals_path = get_detector_postprocessed_signals_path(detector)
    signals = load_h5(signals_path, H5_DATASET_NAME)
    return signals


def get_detector_training_data_path(detector: int):
    return os.path.join(FRAC_SIGNALS_DIR, f'detector_{detector}.h5')


def get_detector_training_data(detector: int):
    data_path = get_detector_training_data_path(detector)
    return load_h5(data_path, H5_DATASET_NAME)


def get_attributes_df(data_path=SPACAL_DATA_PATH):
    """
    :param data_path:
    :return: df, where each column is an attribute
    """
    df = pd.read_parquet(data_path)
    return df


def _calculate_step(df, signal_dim):
    min_timestamp = min(df['timestamp'])  # TODO: (@whiteRa2bit, 2020-09-22) Replace with config params
    max_timestamp = max(df['timestamp'])
    step = (max_timestamp - min_timestamp) / signal_dim
    return step


def _get_step_energies(df, min_timestamp, max_timestamp):
    step_df = df[df['timestamp'] > min_timestamp]
    step_df = step_df[step_df['timestamp'] < max_timestamp]
    step_energies = step_df['PhotonEnergy'].values
    return step_energies


def generate_one_signal(df, signal_dim: int = SIGNAL_DIM, use_postprocessing: bool = False, frac: float = 1.0):
    """
    Generates one output for given df
    :param df: df with info of given detector and event
    :param signal_dim: number of timestamps by which time will be splitted
    :return: np.array [signal_dim] with energies
    """
    if df.empty:
        return np.zeros(signal_dim)

    step = _calculate_step(df, signal_dim)
    steps_energy = []
    for i in range(signal_dim):
        step_energies = _get_step_energies(df, i * step, (i + 1) * step)
        step_energies = np.random.choice(step_energies, int(len(step_energies) * frac))
        step_energy = np.sum(step_energies)
        steps_energy.append(step_energy)
    if use_postprocessing:
        steps_energy = postprocess_signal(steps_energy)

    return np.array(steps_energy)


def generate_signals(df,
                     data_size: int,
                     use_postprocessing: bool = False,
                     signal_dim: int = SIGNAL_DIM,
                     frac: float = 1.0):
    """
    Generates data for a given detector
    :param df_full: pandas df, output of get_events_df()
    :param data_size: number of samples to get
    :param use_postprocessing: whether to use output before or after photodetector
    :param signal_dim: number of timestamps by which time will be splitted
    :param sample_coef: percent of data to take for each step
    :return: np.array with generated events
    """
    if df.empty:
        return np.zeros((data_size, signal_dim))

    step = _calculate_step(df, signal_dim)
    steps_energies = []
    for i in range(signal_dim):
        step_energies = _get_step_energies(df, i * step, (i + 1) * step)
        steps_energies.append(step_energies)

    output_signals = np.zeros((data_size, signal_dim))
    for signal_idx in range(data_size):
        for step_idx, step_energies in enumerate(steps_energies):
            step_energies = np.random.choice(step_energies, int(len(step_energies) * frac))
            step_energy = np.sum(step_energies)
            output_signals[signal_idx][step_idx] = step_energy

    # TODO: (@whiteRa2bit, 2020-09-22) Add postprocessing
    return output_signals


def postprocess_signal(signal):
    """
    Getting result signal after photodetector
    :param signal: Output from generate_one_signal
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
