import functools
import multiprocessing as mp

import numpy as np
import tqdm
import h5py

from generation.config import TRAINING_DATA_DIR, REPEAT_COEF, FRAC_COEF, H5_DATASET_NAME
from generation.dataset.data_utils import save_h5, get_attributes_df, get_event_detector_df, generate_signals, get_detector_training_data_path

_PROCESSORS_NUM = 16


def _process_event_detector(event, detector, n_signals=REPEAT_COEF, frac_coef=FRAC_COEF):
    event_detector_df = get_event_detector_df(event, detector)
    event_signals = generate_signals(event_detector_df, n_signals, frac=frac_coef)
    return event_signals


def _save_detector_signals(detector_signals, detector):
    data_path = get_detector_training_data_path(detector)
    save_h5(detector_signals, H5_DATASET_NAME, data_path)


def main():  # TODO: (@whiteRa2bit, 2020-10-14) Create training_data folder
    df_full = get_attributes_df()
    events = sorted(df_full['event'].unique())
    detectors = sorted(df_full['detector'].unique())

    with mp.Pool(_PROCESSORS_NUM) as pool:
        for detector in detectors:
            print(f"Processing detector {detector}...")
            processing = functools.partial(_process_event_detector, detector=detector)
            detector_signals = list(tqdm.tqdm(pool.imap(processing, events), total=len(events)))

            detector_signals = np.array(detector_signals)
            signal_size = detector_signals.shape[-1]
            detector_signals = detector_signals.reshape(-1, signal_size)
            _save_detector_signals(detector_signals, detector)


if __name__ == '__main__':
    main()
