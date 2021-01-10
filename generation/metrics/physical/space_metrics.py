import numpy as np
import matplotlib.pyplot as plt
import wandb

from generation.metrics.physical.utils import calculate_2d_distributions_distance


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


def _get_space_values(signals):
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


def get_space_metrics_dict(real_signals, fake_signals):
    real_space_values = _get_space_values(real_signals)
    fake_space_values = _get_space_values(fake_signals)
    space_fig = _get_space_fig(real_space_values, fake_space_values)
    space_dist = calculate_2d_distributions_distance(real_space_values, fake_space_values)
    space_metrics_dict = {"Space figure": wandb.Image(space_fig), "Space distance": space_dist}
    return space_metrics_dict
