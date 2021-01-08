import numpy as np
import matplotlib.pyplot as plt
import tqdm

from generation.dataset.data_utils import postprocess_signal

_BINS_NUM = 20


def get_energy_values(signals):
    """
    Returns energy characteristic values for a given set of signals.
    Energy characteristic is a ratio of a signal energy to an amplitude
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: energy characteristic values
    """
    amplitudes = np.max(signals, axis=2)
    energies = np.sum(signals, axis=2)
    ratios = energies / amplitudes
    return ratios


def _calculate_centre_mass(amplitudes):
    """
    Returns centre mass for a given amplitudes array
    :param amplitudes: np array with shape [detectors_num, signals_num]
    :returns: mass centres array with shape [signals_num]
    """
    coords = np.array([[-1, 1], [0, 1], [1, 1], \
          [-1, 0], [0, 0], [1, 0], \
          [-1, -1], [0, -1], [1, -1]])
    return coords.T @ amplitudes


def get_space_values(signals):
    """
    Returns space characteristic values for a given set of signals.
    Space characteristic is a centre mass of detector coordinates,
    where weights are corresping amplitudes
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: space characteristic values
    """
    amplitudes = np.max(signals, axis=2)
    mass_centres = _calculate_centre_mass(amplitudes)
    return mass_centres


def _get_space_fig(real_mass_centres, fake_mass_centres):
    """
    Returns a figure with real and fake mass centres distributions
    :param real_mass_centres: np array with shape [detectors_num, 2]
    :param fake_mass_centres: np array with shape [detectors_num, 2]
    :returns: figure with distributions
    """
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(real_mass_centres[0, :], real_mass_centres[1, :])
    ax.scatter(fake_mass_centres[0, :], fake_mass_centres[1, :])
    ax.legend(['Real', 'Fake'])
    return fig


def _get_time_prediction(signal):
    half_amplitude = np.min(signal) + (np.max(signal) - np.min(signal)) / 2
    for idx, cur_amplitude in enumerate(signal):
        if cur_amplitude > half_amplitude:
            return idx


def get_time_values(signals):
    """
    Returns time characteristic values for a given set of signals.
    Time characteristic is a signal reference time. 
    :param signals: signals np array of shape [detectors_num, signals_num, signal_size]
    :returns: time characteristic values
    """
    postprocessed_signals = [[postprocess_signal(signal) for signal in detector_signals] for detector_signals in signals]
    time_preds = [[_get_time_prediction(signal) for signal in detector_signals] for detector_signals in postprocessed_signals]
    time_preds = np.array(time_preds)
    return time_preds


def _transform_signals(signals_tensor):
    """
    Transforms torch signals tensor to np array and reshapes it
    :param signals_tensor: torch tensor with shape [batch_size, detectors_num, x_dim]
    :returns: np array with shape [detectors_num, batch_size, x_dim]
    """
    signals_array = signals_tensor.cpu().detach().numpy()
    signals_array = np.transpose(signals_array, (1, 0, 2))
    return signals_array


def _get_distributions_fig(real_values, fake_values, bins_num=_BINS_NUM):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(9):  # TODO: (@whiteRa2bit, 2021-01-05) Replace with config constant
        real_detector_values = real_values[i]
        fake_detector_values = fake_values[i] 
        bins = np.histogram(np.hstack((real_detector_values, fake_detector_values)), bins=bins_num)[1]
        ax[i // 3][i % 3].hist(real_detector_values, bins=bins, alpha=0.6)
        ax[i // 3][i % 3].hist(fake_detector_values, bins=bins, alpha=0.6)            
        ax[i // 3][i % 3].legend(["Real", "Fake"])
    return fig


def get_physical_figs(real_signals_tensor, fake_signals_tensor):
    real_signals = _transform_signals(real_signals_tensor)
    fake_signals = _transform_signals(fake_signals_tensor)

    real_energy_values = get_energy_values(real_signals)
    fake_energy_values = get_energy_values(fake_signals)
    energy_fig = _get_distributions_fig(real_energy_values, fake_energy_values)

    real_time_values = get_time_values(real_signals)
    fake_time_values = get_time_values(fake_signals)
    time_fig = _get_distributions_fig(real_time_values, fake_time_values)

    real_space_values = get_space_values(real_signals)
    fake_space_values = get_space_values(fake_signals)
    space_fig = _get_space_fig(real_space_values, fake_space_values)

    return energy_fig, time_fig, space_fig
