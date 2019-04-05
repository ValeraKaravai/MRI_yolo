import os
import sys
import logging

ROOT = os.getcwd()
sys.path.append(ROOT)

DIR_FILES = 'data/descr/'
DIR_IMG = 'data/images/'
DIR_DARKNET = 'darknet/'

PATH_FILES = os.path.join(ROOT, DIR_FILES)
PATH_IMG = os.path.join(ROOT, DIR_IMG)
PATH_DARKNET = os.path.join(ROOT, DIR_DARKNET)

CFG_YOLO = os.path.join(PATH_DARKNET, 'cfg/yolo.cfg')
DATA_YOLO = os.path.join(PATH_DARKNET, 'data/obj.data')
WEIGHT_YOLO = os.path.join(PATH_DARKNET, 'backup/yolo.weights')
pic = os.path.join(ROOT, 'data/img_00122.jpg')

IMG_SIZE = 384


COLUMNS = ['id', 'label', 'file', 'type_disk',
           'height', 'width', 'x', 'y', 'x_center', 'y_center']

COLUMNS_YOLO = ['label', 'x_center',
                'y_center', 'width', 'height']

COLORS = {'1': 'r', '2': 'b', '3': 'g'}

## Logging config
LOG_LEVEL = logging.INFO
LOG_FORMAT = '[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
