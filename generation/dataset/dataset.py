import argparse
import time
import os

import numpy as np
import pandas as pd
import tqdm

from generation.config import DF_DIR, SIGNAL_DIR, \
                PROCESSING_TIME_NORM_COEF, STEPS_NUM, SPACAL_DATA_PATH


def _get_event_dir(base_dir: str, event: int):
    return os.path.join(base_dir, 'event_{}').format(event)


def get_detector_event_df_path(detector: int, event: int, df_dir: str = DF_DIR):
    event_dir = _get_event_dir(df_dir, event)
    df_path = os.path.join(event_dir, 'detector_{}.csv').format(detector)
    return df_path


def get_detector_event_df(detector: int, event: int):
    try:
        df_path = get_detector_event_df_path(detector, event)
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        df = pd.DataFrame({})
    return df


def get_detector_event_signal_path(detector: int, event: int, signal_dir: str = SIGNAL_DIR):
    event_dir = _get_event_dir(signal_dir, event)
    signal_path = os.path.join(event_dir, 'detector_{}.npy').format(detector)
    return signal_path


def get_detector_event_signal(detector: int, event: int):
    try:
        signal_path = get_detector_event_signal_path(detector, event)
        signal = np.load(signal_path)
    except FileNotFoundError:
        signal = np.zeros(STEPS_NUM)
    return signal


def get_attributes_df(data_path=SPACAL_DATA_PATH):
    """
    :param data_path:
    :return: df, where each column is an attribute
    """
    df = pd.read_pickle(data_path)
    return df

def generate_one_signal(df, steps_num: int = STEPS_NUM, sample_coef: float = 1.0):
    """
    Generates one output for given df
    :param df: df with info of given detector and event
    :param steps_num: number of timestamps by which time will be splitted
    :return: np.array [steps_num] with energies
    """
    min_timestamp = min(df['timestamp'])
    max_timestamp = max(df['timestamp'])
    step = (max_timestamp - min_timestamp) / steps_num

    step_energies = []
    for i in range(steps_num):
        step_df = df[df['timestamp'] > i * step]
        step_df = step_df[step_df['timestamp'] < (i + 1) * step]
        step_df = step_df.sample(sample_coef)
        step_energy = sum(step_df['energy'])
        step_energies.append(step_energy)

    return np.array(step_energies)


def generate_signals(df, data_size: int,  use_postprocessing: bool, steps_num: int = STEPS_NUM, sample_coef: float = 1.0):
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
