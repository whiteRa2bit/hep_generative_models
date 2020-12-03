import numpy as np
import torch
from torch.utils.data import Dataset

from generation.config import SIGNAL_SIZE, DETECTORS
from generation.dataset.data_utils import get_detector_training_data


class SignalsDataset(Dataset):
    def __init__(self, detectors=DETECTORS, signal_size=SIGNAL_SIZE, freq=1):
        self.detectors = detectors
        self.signal_size = signal_size
        self.freq = freq
        self.signals = self._get_signals()
        self.noises = self._get_noises()

    def __len__(self):
        return self.noises.shape[1]

    def __getitem__(self, idx):
        noise_tensor = torch.from_numpy(self.noises[:, idx])
        return noise_tensor.float()

    def _get_signals(self):
        signals = []
        for detector in self.detectors:
            signals.append(get_detector_training_data(detector))

        signals = np.array(signals)[:, :, :self.signal_size]
        max_amplitudes = np.max(signals, axis=(1, 2))[:, None, None]
        signals = signals / max_amplitudes
        signals = signals[:, :, :self.signal_size * self.freq:self.freq]
        return signals

    def _get_noises(self):
        mean_signals = np.mean(self.signals, axis=1)[:, None, :]
        noises = self.signals - mean_signals
        max_noises = np.max(noises, axis=(1, 2))[:, None, None]
        min_noises = np.min(noises, axis=(1, 2))[:, None, None]
        noises = (noises - min_noises) / (max_noises - min_noises)
        return noises
