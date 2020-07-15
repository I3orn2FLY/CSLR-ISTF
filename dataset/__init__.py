from dataset.gr import GR_dataset
from dataset.end2end_base import End2EndDataset
from dataset.end2end_img_feat import End2EndImgFeatDataset
from dataset.end2end_stf import End2EndSTFDataset
from dataset.end2end_raw import End2EndRawDataset

from config import *


def get_end2end_datasets(model, vocab, include_test=False):
    if model.use_st_feat or model.use_img_feat:
        batch_size = END2END_STF_BATCH_SIZE
    else:
        batch_size = END2END_RAW_BATCH_SIZE

    args = {"vocab": vocab, "split": "train", "max_batch_size": batch_size,
            "augment_temp": END2END_DATA_AUG_TEMP, "augment_frame": END2END_DATA_AUG_FRAME}

    if model.use_st_feat:
        dataset_class = End2EndSTFDataset
    elif model.use_img_feat:
        dataset_class = End2EndImgFeatDataset
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


def get_gr_datasets(load=False, batch_size=GR_BATCH_SIZE):
    datasets = dict()
    datasets["Train"] = GR_dataset("train", load, batch_size)
    datasets["Val"] = GR_dataset("val", load, batch_size)

    return datasets
