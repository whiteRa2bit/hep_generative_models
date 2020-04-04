import torch
from torch.utils.data import Dataset


class SignalsDataset(Dataset):
    def __init__(self, signals_data):
        self.signals_data = signals_data

    def __len__(self):
        return len(self.signals_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.signals_data[idx].astype("float32")
