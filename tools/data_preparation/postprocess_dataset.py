import argparse
import multiprocessing as mp

import tqdm
import numpy as np
from loguru import logger

from generation.config import POSTPROCESSED_SIGNALS_PATH
from generation.dataset.data_utils import postprocess_signal
from generation.dataset.signals_dataset import SignalsDataset

_NUM_WORKERS = 24
_SIGNALS_LIMIT = 250


def main():
    dataset = SignalsDataset()
    signals = dataset.signals
    detectors_num = signals.shape[0]

    postprocessed_signals = []
    for detector in range(detectors_num):
        logger.info(f"Processing detector {detector}")
        detector_signals = signals[detector][:_SIGNALS_LIMIT]
        with mp.Pool(_NUM_WORKERS) as pool:
            postprocessed_detector_signals = list(tqdm.tqdm(pool.imap(postprocess_signal, detector_signals), total=len(detector_signals)))
        postprocessed_signals.append(postprocessed_detector_signals)

    postprocessed_signals = np.array(postprocessed_signals)
    np.save(POSTPROCESSED_SIGNALS_PATH, postprocessed_signals)


if __name__ == '__main__':
    main()
