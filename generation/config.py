import os
import collections

# Data params
_ENERGY = 1
DATA_DIR = f'/mnt/pafakanov/hep_data/{_ENERGY}GeV/'
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints/')
DF_DIR = os.path.join(DATA_DIR, 'dfs')
FULL_SIGNALS_DIR = os.path.join(DATA_DIR, 'full_signals')
FRAC_SIGNALS_DIR = os.path.join(DATA_DIR, 'fractional_signals')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
EVENTS_PATH = os.path.join(DATA_DIR, "events.npy")
DETECTORS_PATH = os.path.join(DATA_DIR, "detectors.npy")
POSTPROCESSED_SIGNALS_PATH = os.path.join(DATA_DIR, 'postprocessed_signals.npy')
H5_DATASET_NAME = "data"
CONFIG_NAME = "config.json"

# Root params
ROOT_FILES_DIR = os.path.join(RAW_DATA_DIR, 'root_files')
ROOT_TREE_NAME = 'hybrid'
EVENT_ATTR = 'event'
ATTRIBUTES = ['x', 'y', 'z', 'PhotonEnergy', 'detector', 'timestamp',
              'event']  # TODO: (@whiteRa2bit, 2020-08-25) Add namedtuple
INT_ATTRIBUTES = ['event', 'detector']
SPACAL_DATA_PATH = os.path.join(RAW_DATA_DIR, 'particles.parquet')

# Models params
AMPLITUDES_MODEL_NAME = "amplitudes_model"
IMAGES_MODEL_NAME = "images_model"
SHAPES_MODEL_NAME = "shapes_model"
SIGNALS_MODEL_NAME = "signals_model"

# Processing params
SIGNAL_DIM = 2048
PROCESSING_TIME_NORM_COEF = 50
REPEAT_COEF = 3
DETECTORS = range(9)
FRAC_COEF = 0.7
FIG_SIZE = 1

# Training params
WANDB_PROJECT = "hep_generative_models"
RANDOM_SEED = 42
SIGNALS_TRAINING_CONFIG = {
    "g_lr": 3e-4,
    "d_lr": 1e-4,
    "epochs_num": 1000,
    "batch_size": 32,
    "log_each": 5,
    "decay_epoch": 0,
    "save_each": 2,
    "device": "cuda:1",
    "x_dim": 128,
    "x_freq": 2,
    "z_dim": 64,
    "d_coef": 5,
    "use_gp": True,
    "clip_value": 0.01,
    "lambda": 10,
    "channels": 64,
    "pad_size": 9,
    "g_use_scheduler": False,
    "g_lr_multiplier": 5,
    "g_lr_total_epoch": 200,
    "d_use_scheduler": False,
    "d_lr_multiplier": 5,
    "d_lr_total_epoch": 200,
    "g_beta1": 0,
    "g_beta2": 0.99,
    "d_beta1": 0,
    "d_beta2": 0.99
}
SHAPES_TRAINING_CONFIG = {
    "detector": 0,
    "g_lr": 3e-4,
    "d_lr": 1e-4,
    "epochs_num": 1000,
    "batch_size": 32,
    "log_each": 5,
    "decay_epoch": 0,
    "save_each": 2,
    "device": "cuda:1",
    "x_dim": 1024,
    "x_freq": 2,
    "z_dim": 64,
    "d_coef": 5,
    "use_gp": True,
    "clip_value": 0.01,
    "lambda": 10,
    "channels": 64,
    "pad_size": 9,
    "g_use_scheduler": False,
    "g_lr_multiplier": 5,
    "g_lr_total_epoch": 200,
    "d_use_scheduler": False,
    "d_lr_multiplier": 5,
    "d_lr_total_epoch": 200,
    "g_beta1": 0,
    "g_beta2": 0.99,
    "d_beta1": 0,
    "d_beta2": 0.99
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
