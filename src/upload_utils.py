import os
import sys
import shutil
import random
import logging
import requests
import xmltodict
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile

import constants as const

logging.basicConfig(format=const.LOG_FORMAT,
                    level=const.LOG_LEVEL,
                    stream=sys.stdout)


def load_data(from_dir):
    '''

    :param from_dir: dir with file of mri
    :type from_dir: list
    :return: data frame of mri data (from all data)

    '''

    spine_path = [os.path.join(from_dir, f) for f in os.listdir(from_dir)]
    df_raw_list = []
    logging.info('Load from {} {} files'.format(from_dir,
                                                len(spine_path)))
    for file in spine_path:
        df = pd.read_csv(file)
        df_raw_list.append(df)

    logging.info('Concat list of data frames')
    df_raw = pd.concat(df_raw_list,
                       ignore_index=True)

    logging.info('Finish read df = {}'.format(df_raw.shape))
    return df_raw


def polygon_convers(j):
    '''

    :param j: dictionary of polygon
    :type j: dict
    :return x1: coordinare x (left corner)
    :return x1: int
    :return y1: coordinare y (left corner)
    :return y1: int
    :return x_center: coordinare x (center of rectangle)
    :return x_center: int
    :return y_center: coordinare y (center of rectangle)
    :return y_center: int

    '''

    polygon = [list(map(int, i.values())) for i in j['pt']]
    x1 = min(polygon)[0]
    y1 = min(polygon)[1]
    x2 = max(polygon)[0]
    y2 = max(polygon)[1]
    height = y2 - y1
    weight = x2 - x1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    return x1, y1, x_center, y_center, weight, height


def parse_xml(df):
    '''

    :param df: data frame with xml column
    :type df: DataFrame
    :return: DataFrame with new column (from XML)
    '''

    logging.info('Start preprocessing data, shape = {}'.format(df.shape))

    logging.info('Clean data (visualize disk)')
    df = df[df['На срезе визуализируются межпозвоночные диски'] == 'Визуализируются (можно размечать)']

    logging.info('Drop NaN columns of XML')
    df = df.dropna(subset=['XML'])

    list_df_parse = []

    logging.info('Start parse XML, shape = {}'.format(df.shape))
    for i in range(df.shape[0]):

        parse_xml_row = xmltodict.parse(df.XML.iloc[i])
        if 'annotationgroup' in parse_xml_row:
            parse_xml_row = parse_xml_row['annotationgroup']

        tmp_df = pd.DataFrame.from_dict(parse_xml_row['annotation']['object']).drop(['@xmlns', 'attributes', 'type'],
                                                                                    axis=1)
        tmp_df['id'] = df.ID.iloc[i]
        tmp_df['file'] = df.Файлы.iloc[i].replace('/n', '')
        tmp_df['imagesize'] = parse_xml_row['annotation']['imagesize']['nrows'] + \
                              ', ' + \
                              parse_xml_row['annotation']['imagesize']['ncols']
        tmp_df['x'], tmp_df['y'], tmp_df['x_center'], tmp_df['y_center'], \
        tmp_df['width'], tmp_df['height'] = zip(*tmp_df.apply(lambda row: polygon_convers(row['polygon']),
                                                              axis=1))
        tmp_df['parts'] = str(tmp_df['parts'])
        list_df_parse.append(tmp_df)

    df_parse = pd.concat(list_df_parse,
                         ignore_index=True,
                         sort=False)
    df_clean = df_parse.drop(['polygon'], axis=1)

    return df_clean


def parse_type_disk(type_disk_mri):
    '''

    :param type_disk_mri: raw string with type of disk (mri)
    :type type_disk_mri: Series
    :return: Series with parse name of type disk
    '''

    str_type_disk = type_disk_mri.str.split('-', expand=True).iloc[:, 3:5].fillna('')

    return str_type_disk.iloc[:, 0] + str_type_disk.iloc[:, 1]


def preproc_data(df, columns_out, type_disk, cat_type_disk, path_clean_data):
    logging.info('Adding column type_mri')
    df['type_mri'] = df.name.str.split('-', expand=True).iloc[:, 0]

    logging.info('Stats by type_mri {}'.format(df['type_mri'].value_counts().to_dict()))
    logging.info('Stats by deleted {}'.format(df['deleted'].value_counts().to_dict()))
    logging.info('Stats by imagesize {}'.format(df['imagesize'].value_counts().to_dict()))

    logging.info('Filtering data, shape before = {}'.format(df.shape))
    df_filter = df[(df['deleted'] == '0') &
                   (df['type_mri'] == 'shejnyj') &
                   (df['imagesize'] == '384, 384')].copy()

    logging.info('Adding column type of disk')
    df_filter['type_disk'] = parse_type_disk(type_disk_mri=df_filter.name)
    df_filter['type_disk'] = df_filter['type_disk'].map(type_disk)

    logging.info('Adding column label (category)')
    df_filter['label'] = df_filter['type_disk'].map(cat_type_disk)

    df_filter['label'] = df_filter['label'].astype(object)

    logging.info('Filtering data, shape before = {}'.format(df_filter.shape))

    cat_type_disk_int = {v: k for k, v in cat_type_disk.items()}
    logging.info('Category {}'.format(cat_type_disk))
    logging.info('Filter by columns: {}'.format(', '.join(columns_out)))

    df_clean = df_filter[columns_out]

    logging.info('Save clean data to {}, shape = {}'.format(path_clean_data,
                                                            df_clean.shape))
    df_clean.to_csv(path_clean_data,
                    index=False)

    return df_clean, cat_type_disk_int


def main_preprocessing(df, columns_out, type_disk,
                       cat_type_disk,
                       path_clean_data, is_split,
                       path_test_data=None):
    '''

    :param df: data frame for preprocessing (filter of data)
    :type df: DataFrame
    :param columns_out: columns in output df
    :type columns_out: list
    :param type_disk: Dictionary with rename category of disk
    :type type_disk: dict
    :param cat_type_disk
    :param path_clean_data: Path for saving clean data
    :param is_split: bool - split of test and train set or no
    :param path_test_data: path for saving test set
    :return: data frame with categorical variable (label) and without uninformation columns

    '''

    logging.info('Start parse XML column')
    df_parse = parse_xml(df=df)
    df_clean, type_int = preproc_data(df=df_parse,
                                      columns_out=columns_out,
                                      type_disk=type_disk,
                                      cat_type_disk=cat_type_disk,
                                      path_clean_data=path_clean_data)

    imgs = {'main': df_clean['file'].unique()}
    cnt_img = len(imgs['main'])
    cnt_test = int(0.3 * cnt_img)
    if is_split:
        logging.info('Start split test/train set, size = {}'.format(cnt_test))
        test_set, train_set = test_train_split(imgs=imgs['main'],
                                               test_size=cnt_test)
        logging.info('Save test set to = {}'.format(path_clean_data))
        df_clean[df_clean['file'].isin(test_set)].to_csv(path_test_data,
                                                         index=False)

        imgs.update({'test': test_set})
        imgs.update({'train': train_set})

    return df_clean, imgs


def test_train_split(imgs, test_size):
    test_set = random.sample(list(imgs), test_size)
    train_set = set(imgs) - set(test_set)

    return test_set, train_set


def download_file_from_google_drive(id, destination):
    '''

    :param id: id of file from google drive (data zip)
    :param destination: path for download
    :return: None

    '''

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    logging.info('response = {}'.format(response))
    save_response_content(response, destination)

    return


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    chunk_size = 32768

    total_size = int(response.headers.get('content-length', 0))

    logging.info('Start download file, size = {}'.format(total_size))

    with tqdm(desc=destination, total=total_size, unit='B', unit_scale=True) as pbar:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    pbar.update(chunk_size)
                    f.write(chunk)


def unzip_data(zip_file):
    '''

    :param zip_file: name of zip file from google drive
    :type zip_file: str
    :return:
    '''

    logging.info('Unzip = {}'.format(zip_file))
    with ZipFile(zip_file, 'r') as zipObj:
        zipObj.extractall()
    logging.info('Successfully unzip')

    shutil.rmtree('__MACOSX',
                  ignore_errors=True)
