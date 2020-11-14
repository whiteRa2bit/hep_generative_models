import time
from functools import wraps

import torch
import numpy as np
from loguru import logger

from generation.config import RANDOM_SEED


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
