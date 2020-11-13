import multiprocessing as mp
import os

import tqdm
from loguru import logger

from generation.dataset.data_utils import save_h5, get_attributes_df, get_event_dir, get_event_detector_df, \
    get_event_detector_df_path, create_dir
from generation.config import DF_DIR

_PROCESSORS_NUM = 6

_df_full = get_attributes_df()
_df_full.set_index("event", drop=False, inplace=True)
_detectors = sorted(_df_full['detector'].unique())

def _create_dirs(df_dir: str = DF_DIR):
    events = _df_full['event'].unique()
    create_dir(df_dir)
    for event in events:
        create_dir(get_event_dir(df_dir, event))


def _prepare_event_df(event, df_dir: str = DF_DIR):  # TODO: (@whiteRa2bit, 2020-07-21) Add documentation
    event_df = _df_full[_df_full['event'] == event]
    event_df.set_index("detector", drop=False, inplace=True)
    for detector in _detectors:
        df_path = get_event_detector_df_path(event, detector)
        event_detector_df = event_df[event_df['detector'] == detector]
        event_detector_df.to_parquet(df_path, index=False)


def main():
    _create_dirs()
    events = sorted(_df_full["event"].unique())
    # event_dfs = [_df_full[_df_full['event'] == event] for event in events]

    with mp.Pool(_PROCESSORS_NUM) as pool:
        logger.info(f'Preparing events dfs...')
        list(tqdm.tqdm(pool.imap(_prepare_event_df, events), total=len(events)))


if __name__ == '__main__':
    main()
