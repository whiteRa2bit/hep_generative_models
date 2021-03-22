import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import wasserstein_distance

from generation.metrics.physical.utils import calculate_1d_distributions_distances

def get_time_fig(real_times, fake_times):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Times distributions", fontsize=16)

    time_bins = [x for x in np.arange(-0.5, 0.6, 0.1)]

    for i in range(9):
        ax[i // 3][i % 3].set_title(f"Detector {i + 1}")
        ax[i // 3][i % 3].hist(fake_times[:, i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].hist(real_times[:, i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].legend(['Fake', 'Real'])

    return fig


def get_amplitudes_fig(real_amplitudes, fake_amplitudes):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Amplitudes distributions", fontsize=16)

    time_bins = [x for x in np.arange(-0.5, 0.6, 0.1)]

    for i in range(9):
        ax[i // 3][i % 3].set_title(f"Detector {i + 1}")
        ax[i // 3][i % 3].hist(fake_amplitudes[:, i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].hist(real_amplitudes[:, i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].legend(['Fake', 'Real'])

    return fig


def get_amplitude_correlations(amplitudes):
    correlations = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            correlations[i][j] = round(np.corrcoef(amplitudes[i], amplitudes[j])[0][1], 2)
    return correlations


def get_time_aplitudes_figs(real, fake):
    fake_times = np.array([item[:9].cpu().detach().numpy() for item in fake])
    fake_amplitudes = np.array([item[9:].cpu().detach().numpy() for item in fake])

    real_times = np.array([item[:9].cpu().detach().numpy() for item in real])
    real_amplitudes = np.array([item[9:].cpu().detach().numpy() for item in real])

    time_fig = get_time_fig(real_times, fake_times)
    amplitude_fig = get_amplitudes_fig(real_amplitudes, fake_amplitudes)

    time_distances = calculate_1d_distributions_distances(real_times.T, fake_times.T)
    amplitude_distances = calculate_1d_distributions_distances(real_amplitudes.T, fake_amplitudes.T)

    real_amplitude_corrs = get_amplitude_correlations(real_amplitudes)
    fake_amplitude_corrs = get_amplitude_correlations(fake_amplitudes)
    corrs_distance = np.mean(np.abs(real_amplitude_corrs - fake_amplitude_corrs))

    return time_fig, amplitude_fig, time_distances, amplitude_distances, corrs_distance
