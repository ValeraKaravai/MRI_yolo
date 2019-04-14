import pandas as pd


class YoloMetricImg:
    def __init__(self,
                 df_img_origin,
                 df_img_pred,
                 iou_thr=0.5):

        self._img_origin = df_img_origin
        self._img_pred = df_img_pred
        self._iou_thr = iou_thr
        self._pb = df_img_pred.shape[0]
        self._ab = df_img_origin.shape[0]

    def zero_shape(self):
        origin = self._img_origin
        pred = self._img_pred

        result_null = origin

        if origin.shape[0] == 0:
            result_null = pred.copy().reset_index()
        elif pred.shape[0] == 0:
            result_null = origin.copy().reset_index()

        result_null['tp'] = 0
        result_null['iou'] = 0
        result_null['correct'] = False

        result_null = result_null[['file', 'type_disk', 'x', 'y',
                                   'height', 'width',
                                   'iou', 'correct', 'tp']]
        return result_null

    def get_all_iou(self):

        origin = self._img_origin
        pred = self._img_pred

        if (origin.shape[0] == 0) | (pred.shape[0] == 0):
            result_null = self.zero_shape()
            return result_null

        result_list = []
        for row_origin in range(origin.shape[0]):
            box_origin = origin.iloc[row_origin, :]
            ious = []
            corrects = []
            for row_pred in range(pred.shape[0]):
                box_pred = pred.iloc[row_pred, :]

                iou = self.calc_iou(box_a=box_origin,
                                    box_b=box_pred)
                ious.append(iou)

                correct = self.calc_correct(box_a=box_origin,
                                            box_b=box_pred)

                corrects.append(correct)
            iou_c, correct_c = self.get_iou_correct(ious=ious,
                                                    corrects=corrects)
            tp = 1 if (iou_c >= self._iou_thr) & (correct_c) else 0
            result_list.append([box_origin.file,
                                box_origin.type_disk,
                                box_origin.x,
                                box_origin.y,
                                box_origin.height,
                                box_origin.width,
                                iou_c,
                                correct_c,
                                tp])
        result = pd.DataFrame(result_list,
                              columns=['file', 'type_disk', 'x', 'y',
                                       'height', 'width', 'iou', 'correct', 'tp'])

        return result

    def get_tp_img(self, df):
        return sum((df.iou >= self._iou_thr) & df.correct)

    def get_metrics(self):
        result = self.get_all_iou()

        tp = self.get_tp_img(result)

        result_img = {'file': [result.file[0]],
                      'pb': [self._pb],
                      'ab': [self._ab],
                      'tp': [tp]}
        return result, pd.DataFrame.from_dict(result_img)

    @staticmethod
    def get_iou_correct(ious, corrects):

        mx_iou = max(ious)
        ind_mx_iou = ious.index(mx_iou)
        mx_correct = corrects[ind_mx_iou]

        return mx_iou, mx_correct

    @staticmethod
    def calc_correct(box_a,
                     box_b):

        return box_a.type_disk == box_b.type_disk

    @staticmethod
    def calc_iou(box_a, box_b):
        xa, ya, wa, ha = (box_a.x,
                          box_a.y,
                          box_a.width,
                          box_a.height)

        xb, yb, wb, hb = (box_b.x,
                          box_b.y,
                          box_b.width,
                          box_b.height)

        ix = max(0, min(xa + wa, xb + wb) - max(xa, xb))
        iy = max(0, min(ya + ha, yb + hb) - max(ya, yb))

        inter_area = ix * iy

        union_area = wa * ha + wb * hb - inter_area
        return inter_area / union_area
