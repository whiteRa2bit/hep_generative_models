import torch
from torch.utils.data import Dataset
import numpy as np

from generation.config import SIGNAL_DIM
from generation.dataset.data_utils import get_detector_training_data


class Scaler:
    def __init__(self):
        pass

    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)

    def fit_transform(self, data):
        self.min = np.min(data)
        self.max = np.max(data)
        new_data = (data - self.min) / (self.max - self.min)
        return new_data

    def inverse_transform(self, data):
        new_data = data * (self.max - self.min) + self.min
        return new_data


class ShapesDataset(Dataset):
    def __init__(self, detector, signal_dim=SIGNAL_DIM):
        self.signal_dim = signal_dim
        self.detector = detector
        self.signals = self._get_signals()
        self.scaler = Scaler()
        self.noises = self.signals - np.mean(self.signals, axis=0)
        # self.noises = self.scaler.fit_transform(noises)

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, idx):
        item_tensor = torch.from_numpy(self.signals[idx])
        return item_tensor.float()

    def _get_signals(self):
        signals = get_detector_training_data(self.detector)
        signals = self._unify_shape(signals)
        signals = signals[~np.isnan(signals).any(axis=1)]
        return signals[:, :self.signal_dim]

    @staticmethod
    def _unify_shape(data):
        min_values = np.min(data, axis=1)
        max_values = np.max(data, axis=1)
        data = (data - min_values[:, None]) / (max_values - min_values)[:, None]
        return data
