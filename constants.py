import os
import sys
import logging

ROOT = os.getcwd()
sys.path.append(ROOT)

DATA_ZIP = 'data.zip'
DIR_DATA = 'data/'
DIR_FILES = 'descr/'
DIR_IMG = 'images/'
DIR_DARKNET = 'darknet/build/darknet/x64/'

PATH_DATA = os.path.join(ROOT, DIR_DATA)
PATH_IMG = os.path.join(PATH_DATA, DIR_IMG)
PATH_FILES = os.path.join(PATH_DATA, DIR_FILES)
PATH_DARKNET = os.path.join(ROOT, DIR_DARKNET)

CFG_YOLO = os.path.join(PATH_DARKNET, 'cfg/yolo.cfg')
DATA_YOLO = os.path.join(PATH_DARKNET, 'data/obj.data')
WEIGHT_YOLO = os.path.join(PATH_DARKNET, 'backup/yolo_{}.weights')

FILE_ID = '1nlLFYX_Wbcj3IIf1uhtb64Xr1elhsYWj'
IMG_SIZE = 384

COLUMNS = ['id', 'label', 'file', 'type_disk',
           'height', 'width',
           'x', 'y', 'x_center', 'y_center']

COLUMNS_YOLO = ['label', 'x_center',
                'y_center', 'width', 'height']

COLORS = {'patalogical': 'r',
          'suspicion': 'b',
          'healthy': 'g'}

TYPE_DISK = {'patologicheskij': 'patalogical',
             'spodozreniem': 'suspicion',
             'zdorovyj': 'healthy'}

## Logging config
LOG_LEVEL = logging.INFO
LOG_FORMAT = '[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
