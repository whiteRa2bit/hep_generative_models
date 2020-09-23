import os
import collections

# Data params
_ENERGY = 1
DATA_DIR = f'/home/pafakanov/data/hep_data/spacal_simulation/{_ENERGY}GeV/'
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints/')
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
STEPS_NUM = 2048
PROCESSING_TIME_NORM_COEF = 50
REPEAT_COEF = 100
FRAC_COEF = 0.7
FIG_SIZE = 1

# Training params
WANDB_PROJECT = "hep_generative_models"
IMAGES_TRAINING_CONFIG = {
    "detector": 0,
    "lr": 1e-3,
    "epochs_num": 15000,
    "batch_size": 64,
    "log_each": 100,
    "device": "cuda:3",
    "z_dim": 16,
    "disc_coef": 3
}
