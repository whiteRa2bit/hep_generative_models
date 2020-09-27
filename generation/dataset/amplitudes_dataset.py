import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from generation.config import SIGNAL_SIZE, DETECTORS
from generation.dataset.data_utils import get_detector_training_data


class AmplitudesDataset(Dataset):
    def __init__(self, detectors=DETECTORS):
        self.detectors = detectors
        self.origin_amplitudes = self._get_amplitudes()
        self.scaler = MinMaxScaler()
        self.amplitudes = self.scaler.fit_transform(self.origin_amplitudes)

    def _get_amplitudes(self):
        signals = [get_detector_training_data(detector) for detector in self.detectors]
        assert all(len(signals[0]) == len(signals[i]) for i in range(len(signals)))
        amplitudes = np.max(signals, axis=2)
        return amplitudes.T

    def __len__(self):
        return len(self.amplitudes)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.amplitudes[idx])
        return tensor.float()
