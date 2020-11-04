import os
import collections

# Data params
_ENERGY = 1
DATA_DIR = f'/home/pafakanov/data/hep_data/spacal_simulation/{_ENERGY}GeV/'
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints/')
DF_DIR = os.path.join(DATA_DIR, 'dfs')
FULL_SIGNALS_DIR = os.path.join(DATA_DIR, 'full_signals')
TRUNCATED_SIGNALS_DIR = os.path.join(DATA_DIR, 'truncated_signals')
IMAGES_DIR = os.path.join(DATA_DIR, 'fft_images')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
H5_DATASET_NAME = "data"

# Root params
ROOT_FILE_PATH = os.path.join(RAW_DATA_DIR, 'hybrid0.root')
ROOT_TREE_NAME = 'hybrid'
ATTRIBUTES = ['x', 'y', 'z', 'PhotonEnergy', 'detector', 'timestamp',
              'event']  # TODO: (@whiteRa2bit, 2020-08-25) Add namedtuple
INT_ATTRIBUTES = ['event', 'detector']
SPACAL_DATA_PATH = os.path.join(RAW_DATA_DIR, 'particles.parquet')

# Processing params
SIGNAL_SIZE = 2048
PROCESSING_TIME_NORM_COEF = 50
REPEAT_COEF = 20
DETECTORS = range(9)
FRAC_COEF = 0.7
FIG_SIZE = 1

# Training params
WANDB_PROJECT = "hep_generative_models"
RANDOM_SEED = 42
SIGNALS_TRAINING_CONFIG = {
    "g_lr": 3e-4,
    "d_lr": 1e-4,
    "epochs_num": 500,
    "batch_size": 512,
    "log_each": 1,
    "decay_epoch": 0,
    "save_each": 2,
    "device": "cuda:1",
    "x_dim": 512,
    "z_dim": 8,
    "d_coef": 3,
    "lambda": 10,
    "g_use_scheduler": False,
    "g_lr_multiplier": 2,
    "g_lr_total_epoch": 300,
    "d_use_scheduler": True,
    "d_lr_multiplier": 2,
    "d_lr_total_epoch": 300
}
SHAPES_TRAINING_CONFIG = {
    "detector": 0,
    "lr": 1e-5,
    "epochs_num": 5000,
    "batch_size": 256,
    "log_each": 1,
    "save_each": 2,
    "device": "cuda:3",
    "x_dim": 1024,
    "z_dim": 8,
    "disc_coef": 5,
    "lambda": 5
}
AMPLITUDES_TRAINING_CONFIG = {
    "lr": 1e-5,
    "epochs_num": 500,
    "batch_size": 64,
    "log_each": 1,
    "save_each": 2,
    "device": "cuda:1",
    "x_dim": 9,
    "z_dim": 3,
    "disc_coef": 5,
    "lambda": 5
}
IMAGES_TRAINING_CONFIG = {
    "detector": 0,
    "lr": 1e-3,
    "epochs_num": 1000,
    "batch_size": 64,
    "log_each": 1,
    "save_each": 2,
    "device": "cuda:2",
    "z_dim": 16,
    "disc_coef": 3,
    "lambda": 5
}
