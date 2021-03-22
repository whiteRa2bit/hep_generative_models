import numpy as np
import torch
from torch.utils.data import Dataset

from generation.config import SIGNAL_DIM, DETECTORS, POSTPROCESSED_SIGNALS_PATH


class MyDataset(Dataset):
    def __init__(self, detectors=DETECTORS, signal_dim=SIGNAL_DIM, freq=1):
        self.detectors = detectors
        self.signal_dim = signal_dim
        self.freq = freq
        
        self.signals = np.load(POSTPROCESSED_SIGNALS_PATH)       
        self.amplitudes = self._get_amplitudes()
        self.ref_times = self._get_ref_times()
        
    def _get_amplitudes(self):
        amplitudes = np.max(self.signals, axis=2)
        amplitudes /= np.max(amplitudes, axis=1)[:, None]
        return amplitudes - 0.5
    
    def _get_ref_times(self):
        ref_times = np.zeros(self.amplitudes.shape)
        for detector_idx in range(ref_times.shape[0]):
            for signal_idx in range(ref_times.shape[1]):
                ref_times[detector_idx][signal_idx] = self._get_ref_time_pred(self.signals[detector_idx][signal_idx])
        np.nan_to_num(ref_times, 0)
        ref_times /= np.max(ref_times, axis=1)[:, None]
        return ref_times - 0.5
        
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
        
        features = np.concatenate([detector_times, detector_amplitudes], axis=0)
        
        tensor = torch.from_numpy(features)
        return tensor.float()
