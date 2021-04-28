import argparse
import os
import functools
import multiprocessing as mp

import numpy as np
import tqdm
import h5py
from loguru import logger

from generation.utils import timer
from generation.config import FULL_SIGNALS_DIR, FRAC_SIGNALS_DIR, REPEAT_COEF, FRAC_COEF, H5_DATASET_NAME
from generation.dataset.data_utils import save_h5, get_events, get_detectors, get_attributes_df, get_event_detector_df, \
    generate_one_signal, generate_signals, get_detector_signals_path, get_detector_training_data_path, create_dir

_PROCESSORS_NUM = 16
_EVENTS_LIMIT = 500


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--frac', type=float, required=True)
    args = argparser.parse_args()
    return args


def _create_dirs(frac, full_signals_dir: str = FULL_SIGNALS_DIR, frac_signals_dir: str = FRAC_SIGNALS_DIR):
    full_signals_dir = os.path.join(full_signals_dir, str(frac))  # TODO: Remove
    frac_signals_dir = os.path.join(frac_signals_dir, str(frac))  # TODO: Remove
    create_dir(full_signals_dir)
    create_dir(frac_signals_dir)


def _get_event_detector_signals(event, detector, n_signals=REPEAT_COEF, frac_coef=FRAC_COEF):
    event_detector_df = get_event_detector_df(event, detector)
    full_signal = generate_one_signal(event_detector_df)
    frac_signals = generate_signals(event_detector_df, n_signals, frac=frac_coef)
    return full_signal, frac_signals


def _save_detector_signals(full_signals, frac_signals, detector, frac):
    def _modify_path(path):
        path = path.split('/')
        path = path[:6] + [str(frac)] + path[6:]
        return '/'.join(path)

    full_data_path = get_detector_signals_path(detector)
    frac_data_path = get_detector_training_data_path(detector)
    
    full_data_path = _modify_path(full_data_path)  # TODO: Remove
    frac_data_path = _modify_path(frac_data_path)  # TODO: Remove

    save_h5(full_signals, H5_DATASET_NAME, full_data_path)
    save_h5(frac_signals, H5_DATASET_NAME, frac_data_path)


@timer
def main():
    args = parse_args()
    events = get_events()[:_EVENTS_LIMIT]
    detectors = get_detectors()

    _create_dirs(args.frac)


    with mp.Pool(_PROCESSORS_NUM) as pool:
        for detector in detectors:
            logger.info(f"Processing detector {detector}...")
            processing = functools.partial(_get_event_detector_signals, detector=detector, frac_coef=args.frac)
            events_signals = list(tqdm.tqdm(pool.imap(processing, events), total=len(events)))

            full_signals = [event_signals[0] for event_signals in events_signals]
            frac_signals = [event_signals[1] for event_signals in events_signals]

            full_signals = np.array(full_signals)
            frac_signals = np.array(frac_signals)
            signal_dim = frac_signals.shape[-1]
            frac_signals = frac_signals.reshape(-1, signal_dim)
            _save_detector_signals(full_signals, frac_signals, detector, frac=args.frac)


if __name__ == '__main__':
    main()
