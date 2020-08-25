import os
import collections

# Data params
PROJECT_DIR = '/home/pavel/MyDocs/work/lambda/generation/'
DATA_DIR = '/home/pafakanov/data/hep_data/spacal_simulation'

CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, 'checkpoints/')

DF_DIR = os.path.join(DATA_DIR, 'dfs')
SIGNAL_DIR = os.path.join(DATA_DIR, 'signals')
PROCESSED_SIGNAL_DIR = os.path.join(DATA_DIR, 'processed_signals')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')

ATTRIBUTES = ['x', 'y', 'z', 'energy', 'detector', 'timestamp', 'event']  # TODO: (@whiteRa2bit, 2020-08-25) Add namedtuple
ATTRIBUTES_NAMES = collections.namedtuple("Attributes", 'x')
ATTRIBUTE_PATHS = [os.path.join(RAW_DATA_DIR, f'{attr}.txt') for attr in ATTRIBUTES]
SPACAL_DATA_PATH = os.path.join(RAW_DATA_DIR, 'particles.pkl')


# Processing constans
STEPS_NUM = 1024
PROCESSING_TIME_NORM_COEF = 50
