from dataclasses import dataclass
import os
import multiprocessing as mp
import typing as np

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from loguru import logger

from generation.config import IMAGE_DIR, FIG_SIZE
from generation.dataset.data_utils import get_detectors, create_dir
from generation.dataset.shapes_dataset import ShapesDataset, Scaler
from generation.dataset.images_dataset import get_detector_image_dir, get_image_path

_PROCESSORS_NUM = 16


def _create_dirs(detectors, image_dir=IMAGE_DIR):
    create_dir(image_dir)
    for detector in detectors:
        create_dir(get_detector_image_dir(detector))


def _save_image(noise_path):
    noise, path = noise_path
    fig = plt.figure(num=None, figsize=(FIG_SIZE, FIG_SIZE), dpi=100)
    plt.axis('off')
    plt.specgram(noise, Fs=1)
    fig.savefig(path, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()


def _prepare_detector_images(detector):
    shapes_dataset = ShapesDataset(detector)
    noises = shapes_dataset.noises
    paths = [get_image_path(detector, i) for i in range(len(noises))]
    
    with mp.Pool(_PROCESSORS_NUM) as pool:
        list(tqdm.tqdm(pool.imap(_save_image, zip(noises, paths)), total=len(noises)))


def main():
    detectors = get_detectors()
    _create_dirs(detectors)

    for detector in detectors:
        logger.info(f"Processing detector {detector}")
        _prepare_detector_images(detector)


if __name__ == '__main__':
    main()
