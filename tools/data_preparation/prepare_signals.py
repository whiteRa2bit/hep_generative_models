import os
import functools
import multiprocessing as mp

import numpy as np
import tqdm
import h5py
from loguru import logger

from generation.config import FULL_SIGNALS_DIR, TRUNCATED_SIGNALS_DIR, REPEAT_COEF, FRAC_COEF, H5_DATASET_NAME
from generation.dataset.data_utils import save_h5, get_attributes_df, get_event_detector_df, generate_one_signal, \
    generate_signals, get_detector_signals_path, get_detector_training_data_path, create_dir

_PROCESSORS_NUM = 16


def _create_dirs(full_signals_dir: str = FULL_SIGNALS_DIR, trunc_signals_dir: str = TRUNCATED_SIGNALS_DIR):
    create_dir(full_signals_dir)
    create_dir(trunc_signals_dir)


def _process_event_detector(event, detector, n_signals=REPEAT_COEF, frac_coef=FRAC_COEF):
    event_detector_df = get_event_detector_df(event, detector)
    full_signal = generate_one_signal(event_detector_df)
    trunc_signals = generate_signals(event_detector_df, n_signals, frac=frac_coef)
    return full_signal, trunc_signals


def _save_detector_signals(full_signals, trunc_signals, detector):
    full_data_path = get_detector_signals_path(detector)
    trunc_data_path = get_detector_training_data_path(detector)
    save_h5(full_signals, H5_DATASET_NAME, full_data_path)
    save_h5(trunc_signals, H5_DATASET_NAME, trunc_data_path)


def main():  # TODO: (@whiteRa2bit, 2020-10-14) Create training_data folder
    _create_dirs()
    df_full = get_attributes_df()
    events = sorted(df_full['event'].unique())
    detectors = sorted(df_full['detector'].unique())

    with mp.Pool(_PROCESSORS_NUM) as pool:
        for detector in detectors:
            logger.info(f"Processing detector {detector}...")
            processing = functools.partial(_process_event_detector, detector=detector)
            events_signals = list(tqdm.tqdm(pool.imap(processing, events), total=len(events)))

            full_signals = [event_signals[0] for event_signals in events_signals]
            trunc_signals = [event_signals[1] for event_signals in events_signals]

            full_signals = np.array(full_signals)
            trunc_signals = np.array(trunc_signals)
            signal_size = trunc_signals.shape[-1]
            trunc_signals = trunc_signals.reshape(-1, signal_size)
            _save_detector_signals(full_signals, trunc_signals, detector)


if __name__ == '__main__':
    main()
