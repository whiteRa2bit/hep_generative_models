import os
import functools
import multiprocessing as mp

import numpy as np
import tqdm
import h5py
from loguru import logger

from generation.utils import timer
from generation.config import POSTPROCESSED_SIGNALS_DIR, H5_DATASET_NAME
from generation.dataset.data_utils import get_detectors, postprocess_signal, save_h5, get_detector_signals, create_dir, get_detector_postprocessed_signals_path

_PROCESSORS_NUM = 16


def _save_detector_signals(postprocessed_signals, detector):
    postprocessed_data_path = get_detector_postprocessed_signals_path(detector)
    save_h5(postprocessed_signals, H5_DATASET_NAME, postprocessed_data_path)


@timer
def main():
    create_dir(POSTPROCESSED_SIGNALS_DIR)

    detectors = get_detectors()

    with mp.Pool(_PROCESSORS_NUM) as pool:
        for detector in detectors:
            logger.info(f"Processing detector {detector}...")
            signals = get_detector_signals(detector)
            postprocessed_signals = list(tqdm.tqdm(pool.imap(postprocess_signal, signals), total=len(signals)))

            _save_detector_signals(postprocessed_signals, detector)


if __name__ == '__main__':
    main()
