import os

import torch
from torch.utils.data import Dataset
from skimage import io

from generation.config import IMAGES_DIR

class ImageDataset(Dataset):
    def __init__(self, data_dir=IMAGES_DIR):
        self.data_dir = data_dir

    def __len__(self):
        return len(os.listdir(self.data_dir)) - 1

    def __getitem__(self, idx):
        img_path = self.get_image_path(idx)
        img = io.imread(img_path)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor.float() / 255.0
    
    def get_image_path(self, idx):
        return os.path.join(self.data_dir, str(idx)) + '.png'
