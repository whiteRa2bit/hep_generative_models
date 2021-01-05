import numpy as np
import tqdm
from loguru import logger

from generation.dataset.data_utils import postprocess_signal


def get_energy_distribution(signals):
    """
    Returns space characteristic distribution for a given set of signals.
    Space characteristic is a ratio of a signal energy to an amplitude
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: space characteristic distribution
    """
    amplitudes = np.max(signals, axis=2)
    energies = np.sum(signals, axis=2)
    ratios = energies / amplitudes
    return ratios


def get_space_distribution(signals):
    raise ValueError("Not implemented yet")


# TODO: (@whiteRa2bit, 2021-01-05) Use more meaningful variable names
def _get_time_prediction(signal):
    half_amplitude = np.min(signal) + (np.max(signal) - np.min(signal)) / 2
    for idx, item in enumerate(signal):
        if item > half_amplitude:
            return idx


def get_time_distribution(signals):
    """
    Returns time characteristic distribution for a given set of signals.
    Time characteristic is a signal reference time. 
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: time characteristic distribution
    """
    logger.info("Processing signals for time characteristic")
    postprocessed_signals = [postprocess_signal(signal) for signal in tqdm.tqdm(signals)]
    time_preds = [_get_time_prediction(signal) for signal in postprocessed_signals]
    return time_preds


def log_metrics_to_wandb(real_signals, fake_signals):
    def get_distributions_fig(real_distribution, fake_distribution):
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        
        return fig

    real_energy_distribution = get_energy_distribution(real_signals)
    fake_energy_distribution = get_energy_distribution(fake_signals)
    real_time_distribution = get_time_distribution(real_signals)
    fake_time_distribution = get_time_distribution(fake_signals)
    energy_fig =_get_distributions_fig(real_energy_distribution, fake_energy_distribution)
    time_fig = get_distributions_fig(real_time_distribution, fake_time_distribution)
    wandb.log({"generated": fig_gen, "real": fig_real})
