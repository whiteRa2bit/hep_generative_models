import numpy as np
import torch
from torch.utils.data import Dataset

from generation.config import SIGNAL_DIM, DETECTORS
from generation.dataset.data_utils import get_detector_postprocessed_signals


class SignalsDataset(Dataset):
    def __init__(self, detectors=DETECTORS, signal_dim=SIGNAL_DIM, freq=1):
        self.detectors = detectors
        self.signal_dim = signal_dim
        self.freq = freq
        self.signals = self._get_signals()
        print(f'Signals shape: {self.signals.shape}')
        self.noises = self._get_noises()

    def __len__(self):
        return self.noises.shape[1]

    def __getitem__(self, idx):
        item_tensor = torch.from_numpy(self.signals[:, idx])
        return item_tensor.float()

    def _get_signals(self):
        signals = []
        for detector in self.detectors:
            signals.append(get_detector_postprocessed_signals(detector))

        signals = np.array(signals)[:, :, :self.freq * self.signal_dim:self.freq]
        max_amplitudes = np.max(signals, axis=(1, 2))[:, None, None]
        signals = signals / max_amplitudes
        return signals

    def _get_noises(self):
        mean_signals = np.mean(self.signals, axis=1)[:, None, :]
        noises = self.signals - mean_signals
        max_noises = np.max(noises, axis=(1, 2))[:, None, None]
        min_noises = np.min(noises, axis=(1, 2))[:, None, None]
        noises = (noises - min_noises) / (max_noises - min_noises)
        return noises
