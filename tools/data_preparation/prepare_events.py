import functools
import multiprocessing as mp
import os
import time

import pandas as pd
import tqdm

from generation.dataset.data_utils import save_h5, get_attributes_df, _get_event_dir, get_event_detector_df, \
    get_event_detector_df_path, get_event_detector_signal_path, get_event_detector_signal, generate_one_signal
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
        create_dir(_get_event_dir(df_dir, event))
        create_dir(_get_event_dir(signal_dir, event))


def _prepare_event_df(event: int, df_dir: str = DF_DIR):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    detectors = _df_full['detector'].unique()
    event_df = _df_full[_df_full['event'] == event]
    for detector in detectors:
        df_path = get_event_detector_df_path(event, detector)
        event_detector_df = event_df[event_df['detector'] == detector]
        event_detector_df.to_parquet(df_path, index=False)


def _prepare_event_detector_signal(event: int, detector: int, signal_dir: str = SIGNAL_DIR):
    df = get_event_detector_df(event, detector)
    signal_path = get_event_detector_signal_path(event, detector)
    signal = generate_one_signal(df)
    save_h5(signal, H5_DATASET_NAME, signal_path)


def main():
    _create_dirs()
    events = _df_full['event'].unique()
    detectors = _df_full['detector'].unique()

    with mp.Pool(_PROCESSORS_NUM) as pool:
        print(f'Preparing events dfs...')
        list(tqdm.tqdm(pool.imap(_prepare_event_df, events), total=len(events)))

        for detector in detectors:
            print(f'Preparing event signals for detector {detector}')
            processing = functools.partial(_prepare_event_detector_signal, detector=detector)
            list(tqdm.tqdm(pool.imap(processing, events), total=len(events)))


if __name__ == '__main__':
    main()
