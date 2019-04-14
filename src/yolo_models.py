import pandas as pd
from src.yolo_metrics_imgs import YoloMetricsImgs
from src.yolo import YoloPredict


class YoloModels:
    def __init__(self,
                 path_origin,
                 path_pred,
                 test_set,
                 models,
                 cat_type,
                 cfg,
                 data,
                 weights,
                 output_columns):
        self._path_origin = path_origin
        self._path_pred = path_pred
        self._test_set = test_set
        self._models = models
        self._cat_type = cat_type

        self._cfg = cfg
        self._data = data
        self._weights = weights
        self._output_columns = output_columns

    @staticmethod
    def get_df(path):
        return pd.read_csv(path)

    def get_stats_model(self, origin, pred, model):
        metrics_obj = YoloMetricsImgs(df_origin=origin,
                                      df_pred=pred,
                                      classes=list(self._cat_type.keys()),
                                      test_set=self._test_set,
                                      model=model)

        metric_verbose, metric, \
        summary = metrics_obj.get_metrics_by_class(summary=True)

        return metric_verbose, summary

    def get_origin_df_by_model_id(self):
        test_set = self._test_set
        df_origin = self.get_df(self._path_origin)
        df_origin = df_origin[df_origin['file'].isin(test_set)]

        return df_origin

    def get_pred_df_by_model_id(self, model_id):
        name_pred = self._path_pred.format(model_id)
        df_pred = self.get_df(path=name_pred)

        return df_pred

    def get_metrics_models(self):
        df_origin = self.get_origin_df_by_model_id()

        general_metrics_summary_list = []
        general_metric_verbose_list = []

        for model in self._models:

            df_pred = self.get_pred_df_by_model_id(model_id=model)

            metric_verbose, \
            summary = self.get_stats_model(origin=df_origin,
                                           pred=df_pred,
                                           model=model)

            general_metrics_summary_list.append(summary)
            general_metric_verbose_list.append(metric_verbose)

        general_metrics_summary = pd.concat(general_metrics_summary_list)
        general_metric_verbose = pd.concat(general_metric_verbose_list)

        general_metrics_summary.model = general_metrics_summary.model.astype(int)
        general_metrics_summary.f1 = general_metrics_summary.f1.astype(float)
        general_metrics_summary.ap = general_metrics_summary.ap.astype(float)

        return general_metric_verbose, general_metrics_summary

    def get_predict_models(self):
        for model in self._models:
            int_type = {v: k for k, v in self._cat_type.items()}
            yolo_predict = YoloPredict(cfg=self._cfg,
                                       output_columns=self._output_columns,
                                       model=model,
                                       dir_output=self._path_pred.format(model),
                                       write_predict=True,
                                       test_set=self._test_set,
                                       labels=int_type,
                                       data=self._data,
                                       weights=self._weights.format(model))

            df_pred = yolo_predict.get_predictions()

        return



