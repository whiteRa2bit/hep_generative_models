from dataclasses import dataclass
import os
import multiprocessing as mp
import typing as np

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from generation.config import FIG_SIZE
from generation.dataset.data_utils import get_attributes_df
from generation.dataset.signals_dataset import SignalsDataset, Scaler
from generation.dataset.images_dataset import get_image_dir, get_image_path

_PROCESSORS_NUM = 16

@dataclass
class NoiseItem:
    real_noise: np.array
    img_noise: np.array
    path: str


def _create_dir(detector):
    if not os.path.exists(get_image_dir(detector)):
        os.mkdir(get_image_dir(detector))


def _save_image(noise_item):
    fig = plt.figure(num=None, figsize=(FIG_SIZE, FIG_SIZE), dpi=100)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.scatter(noise_item.real_noise, noise_item.img_noise, color='black')
    fig.savefig(noise_item.path, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()


def _prepare_detector_images(detector):
    signals_dataset = SignalsDataset(detector)
    noises = signals_dataset.noises

    fft_noises = np.fft.rfft(noises)
    real_noises = np.real(fft_noises)[:, 1:]  # Explained at fft_shapes notebook
    img_noises = np.imag(fft_noises)[:, 1:]

    real_scaler = Scaler()
    img_scaler = Scaler()
    real_noises = real_scaler.fit_transform(real_noises)
    img_noises = img_scaler.fit_transform(img_noises)

    items = [NoiseItem(real_noises[i], img_noises[i], get_image_path(detector, i)) for i in range(len(real_noises))]
    with mp.Pool(_PROCESSORS_NUM) as pool:
        list(tqdm.tqdm(pool.imap(_save_image, items), total=len(items)))


def main():
    df_full = get_attributes_df()
    detectors = sorted(df_full['detector'].unique())

    for detector in detectors:
        _create_dir(detector)
        print(f"Processing detector {detector}")
        _prepare_detector_images(detector)


if __name__ == '__main__':
    main()
