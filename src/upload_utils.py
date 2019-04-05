import os
import sys
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
                         ignore_index=True)

    return df_parse.drop(['polygon'], axis=1)


def parse_type_disk(type_disk_mri):

    '''

    :param type_disk_mri: raw string with type of disk (mri)
    :type type_disk_mri: Series
    :return: Series with parse name of type disk
    '''

    str_type_disk = type_disk_mri.str.split('-', expand=True).iloc[:, 3:5].fillna('')

    return str_type_disk.iloc[:, 0] + str_type_disk.iloc[:, 1]


def cat_label(str_label):
    '''

    :param str_label: label (type disk)
    :type str_label: Series
    :return: Series with category of type disk (zdarov/patalogiya/podozreniye)
    '''

    cat_name = str_label.astype("category").cat.categories
    cat_name_dict = dict(enumerate(cat_name))

    return str_label.astype("category").cat.codes, \
           cat_name_dict


def preproc_data(df, columns):
    '''

    :param df: data frame for preprocessing (filter of data)
    :type df: DataFrame
    :return: data frame with categorical variable (label) and without uninformation columns

    '''

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

    logging.info('Adding column label (category)')
    df_filter['label'], cat_name_dict = cat_label(str_label=df_filter.type_disk)

    df_filter['label'] = df_filter['label'].astype(object)

    logging.info('Filtering data, shape before = {}'.format(df_filter.shape))
    logging.info('Category {}'.format(cat_name_dict))
    logging.info('Filter by columns: {}'.format(', '.join(columns)))
    return df_filter[columns], cat_name_dict


def download_file_from_google_drive(id, destination):

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    logging.info('response = {}'.format(response))
    save_response_content(response, destination)


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
    # logging.info('Successfully downloaded')


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
