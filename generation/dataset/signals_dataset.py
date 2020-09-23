import torch
from torch.utils.data import Dataset
import numpy as np


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


class SignalsDataset(Dataset):
    def __init__(self, signals):
        signals = self._unify_shape(signals)
        signals = signals[~np.isnan(signals).any(axis=1)]
        noises = signals - np.mean(signals, axis=0)
        self.scaler = Scaler()
        self.signals = signals
        self.noises = self.scaler.fit_transform(noises)

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, idx):
        noise_tensor = torch.from_numpy(self.noises[idx])
        return noise_tensor

    @staticmethod
    def _unify_shape(data):
        min_values = np.min(data, axis=1)
        max_values = np.max(data, axis=1)
        data = (data  - min_values[:, None]) / (max_values - min_values)[:, None]
        return data
