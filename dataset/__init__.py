import sys
import os
from gr import GR_dataset

from end2end_pose import End2EndPoseDataset
from end2end_feat_3d import End2EndTempFusionDataset
from end2end_raw import End2EndRawDataset
from end2end_feat_2d import End2EndImgFeatDataset

sys.path.append(".." + os.sep)

from config import *


def get_end2end_datasets(vocab, include_test=False):
    args = {"vocab": vocab, "split": "train", "max_batch_size": END2END_BATCH_SIZE,
            "augment_temp": END2END_DATA_AUG_TEMP, "augment_frame": END2END_DATA_AUG_FRAME}

    if INP_FEAT:
        if IMG_FEAT_MODEL.startswith("pose"):
            dataset_class = End2EndPoseDataset
        elif IMG_FEAT_MODEL.startswith("densenet121") or IMG_FEAT_MODEL.startswith("googlenet"):
            dataset_class = End2EndImgFeatDataset
        elif IMG_FEAT_MODEL.startswith("resnet{2+1}d"):
            dataset_class = End2EndTempFusionDataset
        else:
            print("Not implemented", IMG_FEAT_MODEL, TEMP_FUSION_TYPE)
            exit(0)
    else:
        dataset_class = End2EndRawDataset
        if TEMP_FUSION_TYPE == 0:
            args["img_size"] = IMG_SIZE_2D
        elif TEMP_FUSION_TYPE == 1:
            args["img_size"] = IMG_SIZE_3D
        else:
            print("Incorrect temp fusion type", TEMP_FUSION_TYPE)
            exit(0)

    tr_dataset = dataset_class(**args)
    args["split"] = "dev"
    val_dataset = dataset_class(**args)

    datasets = {"Train": tr_dataset, "Val": val_dataset}
    if include_test:
        args["split"] = "test"
        datasets["Test"] = dataset_class(**args)

    return datasets


def get_gr_datasets(batch_size=GR_BATCH_SIZE):
    datasets = dict()
    datasets["Train"] = GR_dataset("train", batch_size)
    datasets["Val"] = GR_dataset("val", batch_size)

    return datasets
