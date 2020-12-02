import os
import time
import json
from functools import wraps

import torch
import numpy as np
from loguru import logger

from generation.config import RANDOM_SEED, CHECKPOINT_DIR, CONFIG_NAME


def set_seed(seed=RANDOM_SEED):
    logger.info(f"Set seed {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def _execute_and_profile(fn, *args, **kwargs):
    start_time = time.time()
    fn_result = fn(*args, **kwargs)
    execution_time = time.time() - start_time

    logger.info(f'Elapsed time for "{fn.__name__}": {round(execution_time, 1)} seconds')
    return execution_time, fn_result


def timer(fn):
    """
    Timer decorator. Logs execution time of the function.
    """

    @wraps(fn)
    def _perform(*args, **kwargs):
        _, fn_result = _execute_and_profile(fn, *args, **kwargs)
        return fn_result

    return _perform


def get_checkpoint_dir(run_id):
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, run_id)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def get_config_path(run_id):
    checkpoint_dir = get_checkpoint_dir(run_id)
    config_path = os.path.join(checkpoint_dir, CONFIG_NAME)
    return config_path


def save_as_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)


def read_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
