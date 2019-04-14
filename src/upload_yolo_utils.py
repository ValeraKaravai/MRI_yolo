import os
import sys

import logging
import pandas as pd
from shutil import copy
import constants as const

logging.basicConfig(format=const.LOG_FORMAT,
                    level=const.LOG_LEVEL,
                    stream=sys.stdout)


def yolo_coord_file(df, dir_to, img_name, columns_yolo, img_size):
    img_txt = '{path}/{name}.txt'.format(path=dir_to,
                                         name=img_name.split('.')[0])

    df_yolo = df[df['file'] == img_name][columns_yolo]
    df_yolo[columns_yolo[1:]] = (df_yolo[columns_yolo[1:]] / img_size).round(6)
    df_yolo.to_csv(img_txt,
                   index=False,
                   header=False, sep=' ')


def copy_files(dir_from, dir_to, files):
    logging.info('Copy from {} \n to \n {}'.format(dir_from,
                                                   dir_to))

    list_jpg = [os.path.join(dir_from, f)
                for f in files]
    [copy(f, dir_to) for f in list_jpg]

    logging.info('Success copy {} files'.format(len(files)))

    return


def img_txt(df, dir_to, columns_yolo, img_size, imgs):
    logging.info('Create txt file to dir \n {}'.format(dir_to))
    for img in imgs:
        yolo_coord_file(df=df,
                        dir_to=dir_to,
                        columns_yolo=columns_yolo,
                        img_size=img_size,
                        img_name=img)

    logging.info('Success create {} files txt'.format(len(imgs)))

    return


def test_train_files(test_set, train_set, dir_to_file, dir_to):
    test_img = [dir_to_file + element for element in test_set]
    train_img = [dir_to_file + element for element in train_set]

    logging.info('Write test file {} to \n {}'.format(len(test_img), dir_to))
    pd.DataFrame(test_img).to_csv(dir_to + '/test.txt',
                                  index=False,
                                  header=False,
                                  sep=' ')

    logging.info('Write train file {} to \n {}'.format(len(train_img), dir_to))
    pd.DataFrame(train_img).to_csv(dir_to + '/train.txt',
                                   index=False,
                                   header=False,
                                   sep=' ')

    return test_set, train_set
