import numpy as np
import multiprocessing as mp
import wandb

from generation.dataset.data_utils import postprocess_signal
from generation.metrics.utils import get_detectors_hist_fig, calculate_1d_distributions_distances
from generation.config import DETECTORS

_PROCESSES_NUM = 24


def get_ref_time_pred(signal):
    half_amplitude = np.min(signal) + (np.max(signal) - np.min(signal)) / 2
    for idx, cur_amplitude in enumerate(signal):
        if cur_amplitude > half_amplitude:
            return idx
    return -1


def plot_time_distributions(real_times, fake_times, ax, title, bins):
    """
    :param real_times: [signals_num]
    :param fake_times: [signals_num]
    """    
    ax.set_title(title)
    ax.hist(fake_times, alpha=0.6, bins=bins)
    ax.hist(real_times, alpha=0.6, bins=bins)
    ax.legend(['Fake', 'Real'])
    

def get_time_values(signals, to_postprocess=False):
    """
    Returns time characteristic values for a given set of signals.
    Time characteristic is a signal reference time. 
    :param signals: signals np array of shape [signals_num, signal_size]
    :returns: time characteristic values - np.array with shape [signals_num]
    """
    if to_postprocess:
        with mp.Pool(_PROCESSES_NUM) as pool:
            signals = pool.map(postprocess_signal, signals)
    
    time_values = [get_ref_time_pred(signal) for signal in signals]
    time_values = np.array(time_values)
    return time_values


def get_time_metrics_dict(real_signals, fake_signals):
    real_time_values = get_time_values(real_signals)
    fake_time_values = get_time_values(fake_signals)
    time_fig = get_detectors_hist_fig(real_time_values, fake_time_values)
    time_dists = calculate_1d_distributions_distances(real_time_values, fake_time_values)
    time_metrics_dict = {"Time figure": wandb.Image(time_fig), "Time distance": np.mean(time_dists)}
    return time_metrics_dict
