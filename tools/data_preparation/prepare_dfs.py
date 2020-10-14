import functools
import multiprocessing as mp
import os

import numpy as np
import tqdm

from generation.dataset.data_utils import save_h5, get_attributes_df, get_event_dir, get_event_detector_df, \
    get_event_detector_df_path, create_dir
from generation.config import DF_DIR, H5_DATASET_NAME

_PROCESSORS_NUM = 8
_df_full = get_attributes_df()


def _create_dirs(df_dir: str = DF_DIR):
    events = _df_full['event'].unique()
    create_dir(df_dir)
    for event in events:
        create_dir(get_event_dir(df_dir, event))


def _prepare_event_df(event: int, df_dir: str = DF_DIR):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    detectors = _df_full['detector'].unique()
    event_df = _df_full[_df_full['event'] == event]
    for detector in detectors:
        df_path = get_event_detector_df_path(event, detector)
        event_detector_df = event_df[event_df['detector'] == detector]
        event_detector_df.to_parquet(df_path, index=False)


def main():
    _create_dirs()
    events = sorted(_df_full['event'].unique())
    detectors = sorted(_df_full['detector'].unique())

    with mp.Pool(_PROCESSORS_NUM) as pool:
        print(f'Preparing events dfs...')
        list(tqdm.tqdm(pool.imap(_prepare_event_df, events), total=len(events)))


if __name__ == '__main__':
    main()
