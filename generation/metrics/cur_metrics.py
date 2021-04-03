import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import wasserstein_distance

from generation.metrics.physical.utils import calculate_1d_distributions_distances

def get_time_fig(real_times, fake_times):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Times distributions", fontsize=16)

    time_bins = [x for x in np.arange(0, 1.1, 0.1)]

    for i in range(9):
        ax[i // 3][i % 3].set_title(f"Detector {i + 1}")
        ax[i // 3][i % 3].hist(fake_times[:, i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].hist(real_times[:, i], alpha=0.6, bins=time_bins)
        ax[i // 3][i % 3].legend(['Fake', 'Real'])

    return fig


def get_amplitudes_fig(real_amplitudes, fake_amplitudes):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Amplitudes distributions", fontsize=16)

    time_bins = [x for x in np.arange(0, 1.1, 0.1)]

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


def get_time_aplitudes_figs(real, fake, samples_to_use=256):
    def get_times_amplitudes(items):
        times = np.array([item[:9].cpu().detach().numpy() for item in items])
        amplitudes = np.array([item[9:].cpu().detach().numpy() for item in items])
        return times, amplitudes

    # [N // batch_size, batch_size, 18] -> [batch_size, 18]
    assert len(real) == len(fake)

    real = real.view(-1, real.shape[-1])
    fake = fake.view(-1, fake.shape[-1])

    real_idxs = np.random.randint(high=len(real), size=samples_to_use)
    real_idxs_second = np.random.randint(high=len(real), size=samples_to_use)
    fake_idxs = np.random.randint(high=len(fake), size=samples_to_use)

    real = real[real_idxs]
    real_second = real[real_idxs_second]
    fake = fake[fake_idxs]

    real_times, real_amplitudes = get_times_amplitudes(real)
    real_times_second, real_amplitudes_second = get_times_amplitudes(real_second)
    fake_times, fake_amplitudes = get_times_amplitudes(fake)

    time_fig = get_time_fig(real_times, fake_times)
    amplitude_fig = get_amplitudes_fig(real_amplitudes, fake_amplitudes)

    time_distances = calculate_1d_distributions_distances(real_times.T, fake_times.T)
    normalization_time_distances = calculate_1d_distributions_distances(real_times.T, real_times_second.T)
    amplitude_distances = calculate_1d_distributions_distances(real_amplitudes.T, fake_amplitudes.T)
    normalization_amplitude_distances = calculate_1d_distributions_distances(real_amplitudes.T, real_amplitudes_second.T)

    time_distances /= normalization_time_distances
    amplitude_distances /= normalization_amplitude_distances

    real_amplitude_corrs = get_amplitude_correlations(real_amplitudes)
    fake_amplitude_corrs = get_amplitude_correlations(fake_amplitudes)
    corrs_distance = np.mean(np.abs(real_amplitude_corrs - fake_amplitude_corrs))

    return time_fig, amplitude_fig, time_distances, amplitude_distances, corrs_distance
