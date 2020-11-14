import multiprocessing as mp
import os

import tqdm
import numpy as np
from loguru import logger

from generation.utils import timer
from generation.dataset.data_utils import get_attributes_df, get_event_dir, get_event_detector_df_path, create_dir
from generation.config import DF_DIR, EVENTS_PATH, DETECTORS_PATH

_df_full = get_attributes_df()
_events = sorted(_df_full["event"].unique())
_detectors = sorted(_df_full['detector'].unique())
_df_full = _df_full.groupby("event")


def _create_dirs(df_dir: str = DF_DIR):
    create_dir(df_dir)
    for event in _events:
        create_dir(get_event_dir(df_dir, event))


def _prepare_event_df(event, df_dir: str = DF_DIR):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    event_df = _df_full.get_group(event)
    for detector in _detectors:
        df_path = get_event_detector_df_path(event, detector)
        event_detector_df = event_df[event_df['detector'] == detector]
        event_detector_df.to_parquet(df_path, index=False)


@timer
def main():
    _create_dirs()

    np.save(EVENTS_PATH, _events)
    np.save(DETECTORS_PATH, _detectors)

    logger.info(f'Preparing events dfs...')
    for event in tqdm.tqdm(_events):
        _prepare_event_df(event)


if __name__ == '__main__':
    main()
