import os

PROJECT_DIR = '/home/pavel/MyDocs/work/lambda/generation/'
ROOT_DIR = '/home/pafakanov/data/'

DATA_DIR = os.path.join(ROOT_DIR, 'hep_data/')
EVENTS_DATA = os.path.join(DATA_DIR, 'events')
SPACAL_DATA_DIR = os.path.join(DATA_DIR, 'spacal_simulation')
SPACAL_DATA_PATH = os.path.join(SPACAL_DATA_DIR, 'particles.pkl')
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, 'checkpoints/')
ATTRIBUTES = ['x', 'y', 'z', 'energy', 'detector', 'timestamp', 'event']
ATTRIBUTE_PATHS = [os.path.join(SPACAL_DATA_DIR, f'{attr}.txt') for attr in ATTRIBUTES]

PROCESSING_TIME_NORM_COEF = 50
