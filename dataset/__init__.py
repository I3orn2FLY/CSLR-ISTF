from dataset.gr import GR_dataset
from dataset.end2end_base import End2EndDataset
from dataset.end2end_pose import End2EndPoseDataset
from dataset.end2end_stf import End2EndSTFDataset
from dataset.end2end_raw import End2EndRawDataset

from config import *


def get_end2end_datasets(vocab, use_feat=USE_STF_FEAT, include_test=False):
    if use_feat:
        batch_size = END2END_STF_BATCH_SIZE
    else:
        batch_size = END2END_RAW_BATCH_SIZE

    args = {"vocab": vocab, "split": "train", "max_batch_size": batch_size,
            "augment_temp": END2END_DATA_AUG_TEMP, "augment_frame": END2END_DATA_AUG_FRAME}

    if use_feat:
        if STF_MODEL.startswith("pose"):
            dataset_class = End2EndPoseDataset
        elif STF_MODEL.startswith("resnet{2+1}d"):
            dataset_class = End2EndSTFDataset
        else:
            dataset_class = End2EndDataset
            print("Not implemented", STF_MODEL, STF_TYPE)
            exit(0)
    else:
        dataset_class = End2EndRawDataset

    tr_dataset = dataset_class(**args)
    args["split"] = "dev"
    val_dataset = dataset_class(**args)

    datasets = {"train": tr_dataset, "val": val_dataset}
    if include_test:
        args["split"] = "test"
        datasets["test"] = dataset_class(**args)

    return datasets


def get_gr_datasets(load=True, batch_size=GR_BATCH_SIZE):
    datasets = dict()
    datasets["Train"] = GR_dataset("train", load, batch_size)
    datasets["Val"] = GR_dataset("val", load, batch_size)

    return datasets
