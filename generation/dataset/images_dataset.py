import os

import torch
from torch.utils.data import Dataset
from skimage import io

from generation.config import IMAGE_DIR


class ImagesDataset(Dataset):
    def __init__(self, detector):
        self.detector = detector

    def __len__(self):
        return len(os.listdir(get_detector_image_dir(self.detector)))

    def __getitem__(self, idx):
        img_path = get_image_path(self.detector, idx)
        img = io.imread(img_path)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor.float() / 255.0


def get_image_path(detector, idx):
    return os.path.join(get_detector_image_dir(detector), str(idx)) + '.png'


def get_detector_image_dir(detector):
    return os.path.join(IMAGE_DIR, f"detector_{detector}")
