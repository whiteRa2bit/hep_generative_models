import numpy as np
import matplotlib.pyplot as plt
import tqdm
from loguru import logger

from generation.dataset.data_utils import postprocess_signal

_BINS_NUM = 20


def get_energy_distribution(signals):
    """
    Returns space characteristic distribution for a given set of signals.
    Space characteristic is a ratio of a signal energy to an amplitude
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: space characteristic distribution
    """
    logger.info(f"Signals shape: {signals.shape}")
    logger.info(f"Signals type: {type(signals)}")
    amplitudes = np.max(signals, axis=2)
    energies = np.sum(signals, axis=2)
    ratios = energies / amplitudes
    logger.info(f"Ratios shape: {ratios.shape}")
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
    logger.info(f"Signals shape: {signals.shape}")
    postprocessed_signals = [[postprocess_signal(signal) for signal in detector_signals] for detector_signals in signals]
    time_preds = [[_get_time_prediction(signal) for signal in detector_signals] for detector_signals in postprocessed_signals]
    time_preds = np.array(time_preds)
    logger.info(f"Time preds shape: {time_preds.shape}")
    return time_preds


def get_physical_figs(real_signals_tensor, fake_signals_tensor):
    def transform_signals(signals_tensor):
        """
        Transforms torch signals tensor to np array and reshapes it
        :param signals_tensor: torch tensor with shape [batch_size, detectors_num, x_dim]
        :returns: np array with shape [detectors_num, batch_size, x_dim]
        """
        signals_array = signals_tensor.cpu().detach().numpy()
        signals_array = np.transpose(signals_array, (1, 0, 2))
        return signals_array

    def get_distributions_fig(real_distributions, fake_distributions, bins_num=_BINS_NUM):
        plt.clf()
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(9):  # TODO: (@whiteRa2bit, 2021-01-05) Replace with config constant
            real_detector_distribution = real_distributions[i]
            fake_detector_distribution = fake_distributions[i] 
            bins = np.histogram(np.hstack((real_detector_distribution, fake_detector_distribution)), bins=bins_num)[1]
            ax[i // 3][i % 3].hist(real_detector_distribution, bins=bins)
            ax[i // 3][i % 3].hist(fake_detector_distribution, bins=bins)            
            ax[i // 3][i % 3].legend(["Real", "Fake"])
        return fig

    real_signals = transform_signals(real_signals_tensor)
    fake_signals = transform_signals(fake_signals_tensor)
    real_energy_distribution = get_energy_distribution(real_signals)
    fake_energy_distribution = get_energy_distribution(fake_signals)
    real_time_distribution = get_time_distribution(real_signals)
    fake_time_distribution = get_time_distribution(fake_signals)   
    energy_fig = get_distributions_fig(real_energy_distribution, fake_energy_distribution)
    time_fig = get_distributions_fig(real_time_distribution, fake_time_distribution)
    return energy_fig, time_fig
