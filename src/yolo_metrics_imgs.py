

import sys
import logging
import numpy as np
import pandas as pd
import constants as const
from src.yolo_metrics_img import YoloMetricImg

logging.basicConfig(format=const.LOG_FORMAT,
                    level=const.LOG_LEVEL,
                    stream=sys.stdout)


class YoloMetricsImgs:
    def __init__(self,
                 df_pred,
                 df_origin,
                 test_set,
                 classes,
                 model,
                 iou_thr=0.5):
        self._origin = df_origin
        self._pred = df_pred
        self._test_set = test_set
        self._iou_thr = iou_thr
        self._classes = classes
        self._model = model

    def get_metrics_img(self, by_class=None):
        origin = self._origin
        pred = self._pred

        if by_class is not None:
            origin = origin[origin.type_disk == by_class]
            pred = pred[pred.type_disk == by_class]
        metric_verbose_list = []
        metric_list = []
        for img in self._test_set:
            origin_img = origin[origin.file == img]
            pred_img = pred[pred.file == img]

            if origin_img.shape[0] == 0 & \
                    pred_img.shape[0] == 0:
                continue

            metrics_obj = YoloMetricImg(df_img_origin=origin_img,
                                        df_img_pred=pred_img,
                                        iou_thr=self._iou_thr)

            metric_verbose_img, metric_img = metrics_obj.get_metrics()

            metric_verbose_list.append(metric_verbose_img)
            metric_list.append(metric_img)

        metric_verbose = pd.concat(metric_verbose_list,
                                   axis=0,
                                   sort=True)
        metric = pd.concat(metric_list,
                           axis=0,
                           sort=True)

        metric_cum = self.get_prec_recall_cumulative(metric_verbose)

        ap = self.calc_average_precisions(rec=metric_cum.recall,
                                          prec=metric_cum.precision)
        metric['ap'] = ap
        return metric_cum, metric

    def get_metrics_by_class(self, summary=True):
        metrics_verbose_list = []
        metrics_list = []
        summary_df = None

        logging.info('Metrics for model = {}'.format(self._model))

        for class_id in self._classes:
            metric_verbose_class, metrics_class = self.get_metrics_img(by_class=class_id)

            metric_verbose_class['type_disk'] = class_id
            metrics_class['type_disk'] = class_id

            metrics_verbose_list.append(metric_verbose_class)
            metrics_list.append(metrics_class)

        metric_verbose = pd.concat(metrics_verbose_list,
                                   axis=0,
                                   sort=True)
        metric = pd.concat(metrics_list,
                           axis=0,
                           sort=True)

        metric_verbose['model'] = self._model

        if summary:
            logging.info('Create summary')
            summary_df = self.get_summarize(metric=metric)

        return metric_verbose, metric, summary_df

    def get_summarize(self, metric):

        metric_gr = metric.groupby(by='type_disk').agg({'pb': 'sum',
                                                        'ab': 'sum',
                                                        'tp': 'sum',
                                                        'ap': 'mean'}).reset_index()
        metric_all = metric[['pb', 'tp', 'ab']].sum()
        metric_all['ap'] = metric_gr['ap'].mean()
        metric_all['type_disk'] = 'all'
        metric_all = pd.DataFrame(metric_all).transpose()

        metric_gen = pd.concat([metric_gr, metric_all],
                               axis=0,
                               sort=False)

        metric_gen['precision'] = metric_gen.tp / metric_gen.pb
        metric_gen['recall'] = metric_gen.tp / metric_gen.ab
        metric_gen['f1'] = 2 * metric_gen.precision * metric_gen.recall / \
                           (metric_gen.precision + metric_gen.recall)
        metric_gen['model'] = self._model

        return metric_gen

    @staticmethod
    def get_prec_recall_cumulative(metric):
        metric = metric.sort_values(by='iou',
                                    ascending=False)
        metric['acc_tp'] = metric.tp.cumsum(axis=0)
        metric = metric.reset_index()

        metric['ind'] = metric.index + 1

        cnt = metric.shape[0]

        metric['precision'] = metric.acc_tp / metric.ind
        metric['recall'] = metric.acc_tp / cnt

        return metric

    @staticmethod
    def calc_average_precisions(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
        return ap
