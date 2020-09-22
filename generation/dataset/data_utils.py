import argparse
import time
import os

import numpy as np
import pandas as pd
import tqdm

from generation.config import DF_DIR, SIGNAL_DIR, \
                PROCESSING_TIME_NORM_COEF, STEPS_NUM, SPACAL_DATA_PATH, TRAINING_DATA_DIR


def _get_event_dir(base_dir: str, event: int):
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
    event_dir = _get_event_dir(df_dir, event)
    df_path = os.path.join(event_dir, 'detector_{}.csv').format(detector)
    return df_path


def get_event_detector_df(event: int, detector: int):
    """
    Given detector and event returns corresponding df
    :param event: event number
    :param detector: detector number
    :return: pandas dataframe
    """
    df_path = get_event_detector_df_path(event, detector)
    df = pd.read_csv(df_path)
    return df


def get_event_detector_signal_path(event: int, detector: int, signal_dir: str = SIGNAL_DIR):
    """
    Given detector and event returns path to signal
    :param detector: detector number
    :param event: event number
    :param signal_dir: directory with signal files
    :return: path to np array
    """
    event_dir = _get_event_dir(signal_dir, event)
    signal_path = os.path.join(event_dir, 'detector_{}.npy').format(detector)
    return signal_path


def get_event_detector_signal(event: int, detector: int):
    """
    Given detector and event returns corresponding signal
    :param event: event number
    :param detector: detector number
    :return: numpy array with shape STEPS_NUM
    """
    signal_path = get_event_detector_signal_path(event, detector)
    signal = np.load(signal_path)
    return signal


def get_detector_training_data_path(detector: int):
    return os.path.join(TRAINING_DATA_DIR, f'detector_{detector}.npy')


def get_detector_training_data(detector: int):
    data_path = get_detector_training_data_path(detector)
    with open(data_path, 'rb') as data_file:
        return np.load(data_file)


def get_attributes_df(data_path=SPACAL_DATA_PATH):
    """
    :param data_path:
    :return: df, where each column is an attribute
    """
    df = pd.read_pickle(data_path)
    return df


def _calculate_step(df, steps_num):
    min_timestamp = min(df['timestamp'])  # TODO: (@whiteRa2bit, 2020-09-22) Replace with config params
    max_timestamp = max(df['timestamp'])
    step = (max_timestamp - min_timestamp) / steps_num
    return step


def _get_step_energies(df, min_timestamp, max_timestamp):
    step_df = df[df['timestamp'] > min_timestamp]
    step_df = step_df[step_df['timestamp'] < max_timestamp]
    step_energies = step_df['PhotonEnergy'].values
    return step_energies


def generate_one_signal(df, steps_num: int = STEPS_NUM, use_postprocessing: bool = False, frac: float = 1.0):
    """
    Generates one output for given df
    :param df: df with info of given detector and event
    :param steps_num: number of timestamps by which time will be splitted
    :return: np.array [steps_num] with energies
    """
    if df.empty:
        return np.zeros(steps_num)

    step = _calculate_step(df, steps_num)
    steps_energy = []
    for i in range(steps_num):
        step_energies = _get_step_energies(df, i * step, (i + 1) * step)
        step_energies = np.random.choice(step_energies, int(len(step_energies) * frac))
        step_energy = np.sum(step_energies)
        steps_energy.append(step_energy)
    if use_postprocessing:
        step_energies = postprocess_signal(step_energies)

    return np.array(step_energies)


def generate_signals(df,
                     data_size: int,
                     use_postprocessing: bool = False,
                     steps_num: int = STEPS_NUM,
                     frac: float = 1.0):
    """
    Generates data for a given detector
    :param df_full: pandas df, output of get_events_df()
    :param data_size: number of samples to get
    :param use_postprocessing: whether to use output before or after photodetector
    :param steps_num: number of timestamps by which time will be splitted
    :param sample_coef: percent of data to take for each step
    :return: np.array with generated events
    """
    if df.empty:
        return np.zeros((data_size, steps_num))
    
    step = _calculate_step(df, steps_num)
    steps_energies = []
    for i in range(steps_num):
        step_energies = _get_step_energies(df, i * step, (i + 1) * step)
        steps_energies.append(step_energies)

    output_signals = np.zeros((data_size, steps_num))
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
