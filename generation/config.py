import os
import collections


# Data params
_ENERGY = 1
PROJECT_DIR = '/home/pavel/MyDocs/work/lambda/generation/'
DATA_DIR = f'/home/pafakanov/data/hep_data/spacal_simulation/{_ENERGY}GeV/'


CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, 'checkpoints/')

DF_DIR = os.path.join(DATA_DIR, 'dfs')
SIGNAL_DIR = os.path.join(DATA_DIR, 'signals')
PROCESSED_SIGNAL_DIR = os.path.join(DATA_DIR, 'processed_signals')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')

# Root params
ROOT_FILE_PATH = os.path.join(RAW_DATA_DIR, 'hybrid0.root')
ROOT_TREE_NAME = 'hybrid'

# TODO: (@whiteRa2bit, 2020-08-25) Replace energy with PhotonEnergy
ATTRIBUTES = ['x', 'y', 'z', 'energy', 'detector', 'timestamp', 'event']  # TODO: (@whiteRa2bit, 2020-08-25) Add namedtuple
ATTRIBUTE_PATHS = [os.path.join(RAW_DATA_DIR, f'{attr}.npy') for attr in ATTRIBUTES]
SPACAL_DATA_PATH = os.path.join(RAW_DATA_DIR, 'particles.pkl')


# Processing constans
STEPS_NUM = 1024
PROCESSING_TIME_NORM_COEF = 50
