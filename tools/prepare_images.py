import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from generation.dataset.data_utils import get_event_detector_df, \
get_event_detector_signal, get_event_detector_df_path, \
get_detector_training_data
from generation.dataset.dataset_pytorch import SignalsDataset

_DATA_DIR = '/home/pafakanov/data/hep_data/spacal_simulation/1GeV/fft_images'
_DETECTOR = 0
_PROCESSORS_NUM = 16
SAMPLE_SIZE = 2048


def get_image_path(idx):
    return os.path.join(_DATA_DIR, str(idx)) + '.png'

def save_image(idx):
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.scatter(real_noises[idx], img_noises[idx])
    plt.savefig(get_image_path(idx), bbox_inches='tight', transparent=True, pad_inches=0)
    plt.clf()


def prepare_detector_images(detector):
    

def main():
    origin_data = signals.copy()[:, :SAMPLE_SIZE]
    data = unify_shape(origin_data)
    data = data[~np.isnan(data).any(axis=1)]
    origin_noises = data - np.mean(data, axis=0)
    noises_dataset, noises_scaler, noises = get_dataset(origin_noises)


    fft_noises = np.fft.rfft(noises)
    real_noises = np.real(fft_noises)
    img_noises = np.imag(fft_noises)

    real_noises = real_noises[:, 1:]
    img_noises = img_noises[:, 1:]

    img_scaler = Scaler()
    img_noises = img_scaler.fit_transform(img_noises)

    real_scaler = Scaler()
    real_noises = real_scaler.fit_transform(real_noises)

    idxs = range(len(real_noises))
    with mp.Pool(_PROCESSORS_NUM) as pool:
        list(tqdm.tqdm(pool.imap(save_image, idxs), total=len(idxs)))
