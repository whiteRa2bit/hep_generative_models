import os

import torch
from torch.utils.data import Dataset
from skimage import io

from generation.config import IMAGES_DIR


class ImageDataset(Dataset):
    def __init__(self, detector):
        self.detector = detector

    def __len__(self):
        return len(os.listdir(IMAGES_DIR))

    def __getitem__(self, idx):
        img_path = get_image_path(self.detector, idx)
        img = io.imread(img_path)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor.float() / 255.0


def get_image_path(detector, idx):
    return os.path.join(get_image_dir(detector), str(idx)) + '.png'

def get_image_dir(detector):
    return os.path.join(IMAGES_DIR, f"detector_{detector}")
