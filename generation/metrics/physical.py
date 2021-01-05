import numpy as np
import tqdm
from loguru import logger

from generation.dataset.data_utils import postprocess_signal


def get_energy_characteristic(signals):
    """
    Returns space characteristic for a given set of signals
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: space characteristic
    """
    amplitudes = np.max(signals, axis=2)
    energies = np.sum(signals, axis=2)
    ratios = energies / amplitudes
    return ratios


def get_space_characteristic(signals):
    pass


def _get_time_prediction(signal):
    half_amplitude = np.min(signal) + (np.max(signal) - np.min(signal)) / 2
    prev = signal[0]
    for idx, item in enumerate(signal):
        if item > half_amplitude:
            return idx


def get_time_characteristic(signals):
    logger.info("Processing signals for time characteristic")
    postprocessed_signals = [postprocess_signal(signal) for signal in tqdm.tqdm(signals)]
    time_preds = [_get_time_prediction(signal) for signal in postprocessed_signals]
    return time_preds
