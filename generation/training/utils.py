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
