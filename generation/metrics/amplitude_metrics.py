import matplotlib.pyplot as plt
import numpy as np
import wandb
from loguru import logger
from scipy.stats import wasserstein_distance

from generation.metrics.utils import calculate_1d_distributions_distances, calculate_2d_distributions_distance
from generation.config import DETECTORS


def get_amplitude_fig(real_amplitudes, fake_amplitudes):
    """
    :param real_amplitudes: [detectors_num, signals_num]
    :param fake_amplitudes: [detectors_num, signals_num]
    """
    assert (real_amplitudes.shape == fake_amplitudes.shape)
    assert len(real_amplitudes) == len(DETECTORS)

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Amplitudes distributions", fontsize=16)

    time_bins = [x for x in np.arange(0, 1.1, 0.1)]

    for i in range(9):
        ax[i // 3][i % 3].set_title(f"Detector {i + 1}")
        ax[i // 3][i % 3].hist(fake_amplitudes[i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].hist(real_amplitudes[i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].legend(['Fake', 'Real'])

    return fig


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



def _get_space_fig(real_mass_centres, fake_mass_centres):
    """
    Returns a figure with real and fake mass centres distributions
    :param real_mass_centres: np array with shape [detectors_num, 2]
    :param fake_mass_centres: np array with shape [detectors_num, 2]
    :returns: figure with distributions
    """
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(real_mass_centres[:, 0], real_mass_centres[:, 1])
    ax.scatter(fake_mass_centres[:, 0], fake_mass_centres[:, 1])
    ax.legend(['Real', 'Fake'])
    return fig


def get_space_metrics_dict(real_amplitudes, fake_amplitudes):
    """
    Returns space characteristic values for a given set of signals.
    Space characteristic is a centre mass of detector coordinates,
    where weights are corresping amplitudes
    :param real_amplitudes: signals np array of shape [detectors_num, signals_num]
    :param fake_amplitudes: signals np array of shape [detectors_num, signals_num]
    :returns: space metrics dict
    """
    assert (real_amplitudes.shape == fake_amplitudes.shape)
    assert len(real_amplitudes) == len(DETECTORS)

    real_space_values = _calculate_centre_mass(real_amplitudes)
    fake_space_values = _calculate_centre_mass(fake_amplitudes)
    space_fig = _get_space_fig(real_space_values, fake_space_values)
    space_dist = calculate_2d_distributions_distance(real_space_values, fake_space_values)
    space_metrics_dict = {"Space figure": wandb.Image(space_fig), "Space distance": space_dist}
    return space_metrics_dict
