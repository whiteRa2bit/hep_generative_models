import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

_BINS_NUM = 20


def get_detectors_hist_fig(real_values, fake_values, bins_num=_BINS_NUM):
    """
    Returns a figure with 3 * 3 subplots, each subplot is a corresponding detector
    histograms of real and fake values
    :param real_values: - np.array of shape [detectors_num, signals_num]
    :param fake_values: - np.array of shape [detectors_num, signals_num]
    :return: plt figure with 3 * 3 subplots
    """
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(9):  # TODO: (@whiteRa2bit, 2021-01-05) Replace with config constant
        real_detector_values = real_values[i]
        fake_detector_values = fake_values[i]
        bins = np.histogram(np.hstack((real_detector_values, fake_detector_values)), bins=bins_num)[1]
        ax[i // 3][i % 3].hist(real_detector_values, bins=bins, alpha=0.6)
        ax[i // 3][i % 3].hist(fake_detector_values, bins=bins, alpha=0.6)
        ax[i // 3][i % 3].legend(["Real", "Fake"])
    return fig


def transform_signals(signals_tensor):
    """
    Transforms torch signals tensor to np array and reshapes it
    :param signals_tensor: torch tensor with shape [batch_size, detectors_num, x_dim]
    :returns: np array with shape [detectors_num, batch_size, x_dim]
    """
    signals_array = signals_tensor.cpu().detach().numpy()
    signals_array = np.transpose(signals_array, (1, 0, 2))
    return signals_array


def calculate_1d_distributions_distances(real_values, fake_values):
    """
    Calculates an array of distances between fake and real distributions for each detector
    :param: real_values - np.array with real values of shape [detectors_num, signals_num]
    :param: fake_values - np.array with real values of shape [detectors_num, signals_num]
    :return: distances between fake and real distributions for each detector - np.array [detectors_num]
    """
    assert len(real_values) == len(fake_values)
    distances = []
    for detector_idx in range(len(real_values)):
        detector_real_values = real_values[detector_idx]
        detector_fake_values = fake_values[detector_idx]
        distances.append(wasserstein_distance(detector_real_values, detector_fake_values))
    return distances


def calculate_2d_distributions_distance(real_values, fake_values):
    """
    For details: https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
    :param: real_values - 2d np.array
    :param: fake_values - 2d np.array with the same shape as real_values
    :return: distance between real and fake distributions
    """
    assert len(real_values) == len(fake_values)
    distances = cdist(real_values, fake_values)
    assignment = linear_sum_assignment(distances)
    return distances[assignment].sum() / len(real_values)
