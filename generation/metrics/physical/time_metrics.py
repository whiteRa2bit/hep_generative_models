import numpy as np
import multiprocessing as mp
import wandb

from generation.dataset.data_utils import postprocess_signal
from generation.metrics.physical.utils import get_detectors_hist_fig, calculate_1d_distributions_distances

_PROCESSES_NUM = 24


def _get_ref_time_pred(signal):
    half_amplitude = np.min(signal) + (np.max(signal) - np.min(signal)) / 2
    for idx, cur_amplitude in enumerate(signal):
        if cur_amplitude > half_amplitude:
            return idx


def _get_time_values(signals):
    """
    Returns time characteristic values for a given set of signals.
    Time characteristic is a signal reference time. 
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: time characteristic values - np.array with shape [detectors_num, signals_num]
    """
    postprocessed_signals = []
    with mp.Pool(_PROCESSES_NUM) as pool:
        for detector_signals in signals:
            postprocessed_detector_signals = pool.map(postprocess_signal, detector_signals)
            postprocessed_signals.append(postprocessed_detector_signals)
    time_values = [
        [_get_ref_time_pred(signal) for signal in detector_signals] for detector_signals in postprocessed_signals
    ]
    time_values = np.array(time_values)
    return time_values


def get_time_metrics_dict(real_signals, fake_signals):
    real_time_values = _get_time_values(real_signals)
    fake_time_values = _get_time_values(fake_signals)
    time_fig = get_detectors_hist_fig(real_time_values, fake_time_values)
    time_dists = calculate_1d_distributions_distances(real_time_values, fake_time_values)
    time_metrics_dict = {"Time figure": wandb.Image(time_fig), "Time distance": np.mean(time_dists)}
    return time_metrics_dict
