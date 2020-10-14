import functools
import multiprocessing as mp
import os

import numpy as np
import tqdm

from generation.dataset.data_utils import save_h5, get_attributes_df, get_event_dir, get_event_detector_df, \
    get_event_detector_df_path, get_detector_data_path, get_detector_data, generate_one_signal
from generation.config import DF_DIR, SIGNAL_DIR, ATTRIBUTES, SIGNAL_SIZE, H5_DATASET_NAME

_PROCESSORS_NUM = 8
_df_full = get_attributes_df()


def _create_dirs(df_dir: str = DF_DIR, signal_dir: str = SIGNAL_DIR):
    def create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    events = _df_full['event'].unique()
    create_dir(df_dir)
    create_dir(signal_dir)
    for event in events:
        create_dir(get_event_dir(df_dir, event))
        create_dir(get_event_dir(signal_dir, event))


def _prepare_event_df(event: int, df_dir: str = DF_DIR):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    detectors = _df_full['detector'].unique()
    event_df = _df_full[_df_full['event'] == event]
    for detector in detectors:
        df_path = get_event_detector_df_path(event, detector)
        event_detector_df = event_df[event_df['detector'] == detector]
        event_detector_df.to_parquet(df_path, index=False)


def _save_detector_signals(detector_signals, detector):
    data_path = get_detector_data_path(detector)
    save_h5(detector_signals, H5_DATASET_NAME, data_path)


def _prepare_event_detector_signal(event: int, detector: int, signal_dir: str = SIGNAL_DIR):
    df = get_event_detector_df(event, detector)
    signal = generate_one_signal(df)
    return signal


def main():
    _create_dirs()
    events = sorted(_df_full['event'].unique())
    detectors = sorted(_df_full['detector'].unique())

    with mp.Pool(_PROCESSORS_NUM) as pool:
        # print(f'Preparing events dfs...')
        # list(tqdm.tqdm(pool.imap(_prepare_event_df, events), total=len(events)))

        for detector in detectors:
            print(f'Preparing event signals for detector {detector}')
            processing = functools.partial(_prepare_event_detector_signal, detector=detector)
            detector_signals = list(tqdm.tqdm(pool.imap(processing, events), total=len(events)))
            
            detector_signals = np.array(detector_signals)
            signal_size = detector_signals.shape[-1]
            print(detector_signals.shape)
            detector_signals = detector_signals.reshape(-1, signal_size)
            print(detector_signals.shape)
            _save_detector_signals(detector_signals)

if __name__ == '__main__':
    main()
