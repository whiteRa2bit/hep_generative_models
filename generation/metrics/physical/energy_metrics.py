import numpy as np
import wandb

from generation.metrics.physical.utils import get_detectors_hist_fig, calculate_1d_distributions_distances


def get_energy_values(signals):
    """
    Returns energy characteristic values for a given set of signals.
    Energy characteristic is a ratio of a signal energy to an amplitude
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: energy characteristic values - np.array with shape [detectors_num, signals_num]
    """
    amplitudes = np.max(signals, axis=2)
    energies = np.sum(signals, axis=2)
    ratios = energies / amplitudes
    return ratios


def get_energy_metrics_dict(real_signals, fake_signals):
    real_energy_values = get_energy_values(real_signals)
    fake_energy_values = get_energy_values(fake_signals)
    energy_fig = get_detectors_hist_fig(real_energy_values, fake_energy_values)
    energy_dists = calculate_1d_distributions_distances(real_energy_values, fake_energy_values)
    energy_metrics_dict = {"Energy figure": wandb.Image(energy_fig), "Energy distance": np.mean(energy_dists)}
    return energy_metrics_dict
