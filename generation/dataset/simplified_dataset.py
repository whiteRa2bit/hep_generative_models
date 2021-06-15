from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset

from generation.config import SIGNAL_DIM, DETECTORS
from generation.dataset.data_utils import get_detector_postprocessed_signals
from generation.metrics.time_metrics import get_ref_time_pred


class SimplifiedDataset(Dataset):
    def __init__(self, detectors=DETECTORS, signal_dim=SIGNAL_DIM, freq=1):
        self.detectors = detectors
        self.signal_dim = signal_dim
        self.freq = freq

        self.signals = self._get_signals()
        self.amplitudes = self._get_amplitudes()
        self.ref_times = self._get_ref_times()

    def _get_signals(self):
        signals = []
        for detector in self.detectors:
            signals.append(get_detector_postprocessed_signals(detector))

        signals = np.array(signals)[:, :, :self.freq * self.signal_dim:self.freq]
        max_amplitudes = np.max(signals, axis=(1, 2))[:, None, None]
        signals = signals / max_amplitudes
        return signals

    def _get_amplitudes(self):
        amplitudes = np.max(self.signals, axis=2)
        return amplitudes

    def _get_ref_times(self):
        ref_times = np.zeros(self.amplitudes.shape)
        for detector_idx in range(ref_times.shape[0]):
            for signal_idx in range(ref_times.shape[1]):
                ref_times[detector_idx][signal_idx] = get_ref_time_pred(self.signals[detector_idx][signal_idx])
        np.nan_to_num(ref_times, 0)
        return ref_times

    def __len__(self):
        return self.signals.shape[1]

    def __getitem__(self, idx):
        detector_times = self.ref_times[:, idx]
        detector_amplitudes = self.amplitudes[:, idx]

        features = np.concatenate([detector_times, detector_amplitudes], axis=0)

        tensor = torch.from_numpy(features)
        return tensor.float()
