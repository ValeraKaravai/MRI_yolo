import sys
import os
import logging
import constants as const
from src.yolo_utils import *

logging.basicConfig(format=const.LOG_FORMAT,
                    level=const.LOG_LEVEL,
                    stream=sys.stdout)
os.chdir(const.PATH_DARKNET)


class YoloPredict:
    def __init__(self,
                 model,
                 cfg,
                 data,
                 weights,
                 output_columns,
                 labels,
                 test_set,
                 dir_output,
                 write_predict=True,
                 thr=.2):
        self._cfg = cfg
        self._data = data
        self._weights = weights
        self._output_columns = output_columns
        self._labels = labels
        self._thr = thr
        self._test_set = test_set
        self._model = model
        self._dir_output = dir_output
        self._write_predict = write_predict

    def get_net_meta(self):
        '''
        Function for load out network.
        You need paths to weight, cfg, data

        :return net and meta
        '''

        logging.info('Encode to ascii')
        cfg = self._cfg.encode('ascii')
        weights = self._weights.encode('ascii')
        data = self._data.encode('ascii')

        logging.info('Load net {} {}'.format(cfg, weights))
        net = load_net(cfg, weights, 0)

        logging.info('Load meta {}'.format(data))
        meta = load_meta(data)
        return net, meta

    def get_detection(self, net, meta):
        '''
        Function for detection of boxes
        :param test_set: set of images
        :param net: network (from get_net_meta)
        :param meta: meta data network (from get_net_meta)
        :return DaraFrame of predictions
        '''
        df_pred_list = []
        test_set = self._test_set
        cnt = len(test_set)
        for i in range(cnt):
            results = detect_df(net=net,
                                meta=meta,
                                img=test_set[i],
                                cat_label=self._labels,
                                columns=self._output_columns,
                                threshold=self._thr)
            df_pred_list.append(results)
        df_pred = pd.concat(df_pred_list)

        return df_pred

    def generate_name(self):
        '''

        :param img: name of image with .jpg
        :return:
        '''

        return self._dir_output.format(self._model)

    def write_predict(self, df):

        name_file = self.generate_name()
        logging.info('Write to {}, cnt {}'.format(name_file,
                                                  df.shape))
        df.to_csv(name_file,
                  index=False)

    def get_predictions(self):
        '''
        Main function for predictions and load network
        :return: data frame of predictions
        '''
        logging.info('Start load network')
        net, meta = self.get_net_meta()

        logging.info('Start prediction')
        df_prediction = self.get_detection(net=net, meta=meta)

        if self._write_predict:
            self.write_predict(df=df_prediction)

        return df_prediction
