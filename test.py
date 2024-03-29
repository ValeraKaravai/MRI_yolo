import constants as const

from src.yolo_models import *

test = ['img_00434.jpg',
        'img_00081.jpg',
        'img_01057.jpg',
        'img_00644.jpg',
        'img_01072.jpg',
        'img_00386.jpg',
        'img_01035.jpg',
        'img_01062.jpg',
        'img_01270.jpg',
        'img_00316.jpg',
        'img_01037.jpg',
        'img_00189.jpg',
        'img_00044.jpg',
        'img_00320.jpg',
        'img_00385.jpg',
        'img_00176.jpg',
        'img_00108.jpg',
        'img_01281.jpg',
        'img_01084.jpg',
        'img_00046.jpg',
        'img_00178.jpg',
        'img_00279.jpg',
        'img_01163.jpg',
        'img_00602.jpg',
        'img_01139.jpg',
        'img_01148.jpg',
        'img_00173.jpg',
        'img_01034.jpg',
        'img_01174.jpg',
        'img_00082.jpg',
        'img_00125.jpg',
        'img_00384.jpg',
        'img_00146.jpg',
        'img_01060.jpg',
        'img_01203.jpg',
        'img_01033.jpg',
        'img_01020.jpg',
        'img_01188.jpg',
        'img_01280.jpg',
        'img_00711.jpg',
        'img_00863.jpg',
        'img_01151.jpg',
        'img_01175.jpg',
        'img_00032.jpg',
        'img_00945.jpg',
        'img_00929.jpg',
        'img_00798.jpg',
        'img_01213.jpg',
        'img_01007.jpg',
        'img_00992.jpg',
        'img_01268.jpg',
        'img_00760.jpg',
        'img_00628.jpg',
        'img_00786.jpg',
        'img_00126.jpg',
        'img_00799.jpg',
        'img_01049.jpg',
        'img_00394.jpg',
        'img_00993.jpg',
        'img_01266.jpg',
        'img_01022.jpg',
        'img_01082.jpg',
        'img_01004.jpg',
        'img_00162.jpg',
        'img_00328.jpg',
        'img_00007.jpg',
        'img_00783.jpg',
        'img_00186.jpg',
        'img_00138.jpg',
        'img_00359.jpg',
        'img_01162.jpg',
        'img_01101.jpg',
        'img_00356.jpg',
        'img_00745.jpg',
        'img_00878.jpg',
        'img_00030.jpg',
        'img_01216.jpg',
        'img_01044.jpg',
        'img_00632.jpg',
        'img_00303.jpg',
        'img_00056.jpg',
        'img_01265.jpg',
        'img_00121.jpg',
        'img_01241.jpg',
        'img_00785.jpg',
        'img_01036.jpg',
        'img_00735.jpg',
        'img_00346.jpg',
        'img_00099.jpg',
        'img_01086.jpg',
        'img_00372.jpg',
        'img_00227.jpg',
        'img_00915.jpg',
        'img_00292.jpg',
        'img_00057.jpg',
        'img_00383.jpg',
        'img_00148.jpg',
        'img_00954.jpg',
        'img_01164.jpg',
        'img_00723.jpg',
        'img_00917.jpg']

models_obj = YoloModels(path_origin=const.PATH_DATA + 'data_clean.csv',
                        path_pred=const.PATH_DATA + '{}_prediction.csv',
                        test_set=test,
                        models=['35000'],
                        cat_type=const.CAT_TYPE_DISK,
                        cfg=const.CFG_YOLO,
                        data=const.DATA_YOLO,
                        weights=const.WEIGHT_YOLO,
                        output_columns=const.COLUMNS_YOLO_OUT)

models_obj.get_predict_models()

models_obj.get_metrics_models()
