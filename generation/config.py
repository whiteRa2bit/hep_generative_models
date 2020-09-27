import os
import collections

# Data params
_ENERGY = 1
DATA_DIR = f'/home/pafakanov/data/hep_data/spacal_simulation/{_ENERGY}GeV/'
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints/')
DF_DIR = os.path.join(DATA_DIR, 'dfs')
SIGNAL_DIR = os.path.join(DATA_DIR, 'signals')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training_data')
IMAGES_DIR = os.path.join(DATA_DIR, 'fft_images')

# Root params
ROOT_FILE_PATH = os.path.join(RAW_DATA_DIR, 'hybrid0.root')
ROOT_TREE_NAME = 'hybrid'
ATTRIBUTES = ['x', 'y', 'z', 'PhotonEnergy', 'detector', 'timestamp',
              'event']  # TODO: (@whiteRa2bit, 2020-08-25) Add namedtuple
INT_ATTRIBUTES = ['event', 'detector']
ATTRIBUTE_PATHS = [os.path.join(RAW_DATA_DIR, f'{attr}.npy') for attr in ATTRIBUTES]
SPACAL_DATA_PATH = os.path.join(RAW_DATA_DIR, 'particles.pkl')

# Processing params
SIGNAL_SIZE = 2048
PROCESSING_TIME_NORM_COEF = 50
REPEAT_COEF = 100
DETECTORS = range(9)
FRAC_COEF = 0.7
FIG_SIZE = 1

# Training params
WANDB_PROJECT = "hep_generative_models"
RANDOM_SEED = 42
SIGNALS_TRAINING_CONFIG = {
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
    "lambda": 1
}
AMPLITUDES_TRAINING_CONFIG = {
    "lr": 1e-4,
    "epochs_num": 1000,
    "batch_size": 128,
    "log_each": 1,
    "save_each": 2,
    "device": "cuda:1",
    "x_dim": 9,
    "z_dim": 8,
    "disc_coef": 5,
    "lambda": 3
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
