from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset

from generation.config import SIGNAL_DIM, DETECTORS, POSTPROCESSED_SIGNALS_PATH, POSTPROCESSED_DIR


_FRAC_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class MyDataset(Dataset):
    def __init__(self, detectors=DETECTORS, signal_dim=SIGNAL_DIM, freq=1):
        self.detectors = detectors
        self.signal_dim = signal_dim
        self.freq = freq

        self.signals, self.labels = self._get_signals_and_labels()
        self.amplitudes = self._get_amplitudes()
        self.ref_times = self._get_ref_times()

    def _get_signals_and_labels(self):
        signals = []
        labels = []
        for frac in _FRAC_VALUES:
            cur_signals_path = join(POSTPROCESSED_DIR, f'{frac}.npy')
            cur_signals = np.load(cur_signals_path)  
            signals.append(cur_signals)
            labels += [frac] * cur_signals.shape[1]
        signals = np.concatenate(signals, axis=1)
        labels = np.array(labels)

        return signals, labels


    def _get_amplitudes(self):
        amplitudes = np.max(self.signals, axis=2)
        amplitudes /= np.max(amplitudes, axis=1)[:, None]
        return amplitudes
    
    def _get_ref_times(self):
        ref_times = np.zeros(self.amplitudes.shape)
        for detector_idx in range(ref_times.shape[0]):
            for signal_idx in range(ref_times.shape[1]):
                ref_times[detector_idx][signal_idx] = self._get_ref_time_pred(self.signals[detector_idx][signal_idx])
        np.nan_to_num(ref_times, 0)
        ref_times /= np.max(ref_times, axis=1)[:, None]
        return ref_times
        
    @staticmethod
    def _get_ref_time_pred(signal):
        half_amplitude = np.min(signal) + (np.max(signal) - np.min(signal)) / 2
        for idx, cur_amplitude in enumerate(signal):
            if cur_amplitude > half_amplitude:
                return idx

    def __len__(self):
        return self.signals.shape[1]

    def __getitem__(self, idx):
        detector_times = self.ref_times[:, idx]
        detector_amplitudes = self.amplitudes[:, idx]
        
        features = np.concatenate([[self.labels[idx]], detector_times, detector_amplitudes], axis=0)

        tensor = torch.from_numpy(features)
        return tensor.float()
