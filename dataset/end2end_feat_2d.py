import sys
import os
import torch
import numpy as np

from end2end_base import End2EndDataset, random_skip, down_sample

sys.path.append(".." + os.sep)

from config import *
# TODO test this

class End2EndImgFeatDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not IMG_FEAT_MODEL.startswith("densenet121") or not IMG_FEAT_MODEL.startswith("googlenet"):
            print("Incorrect feat model:", IMG_FEAT_MODEL)
            exit(0)
        super(End2EndImgFeatDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_feat(self, row, glosses=None):
        if SOURCE == "PH":
            feat_path = os.sep.join([VIDEO_FEAT_DIR, self.split, row.folder.replace("/1/*.png", ".pt")])
        elif SOURCE == "KRSL":
            feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")
        else:
            return None, None, None

        if not os.path.exists(feat_path):
            return None, None, None

        feat = torch.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) * 4:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = torch.load(self.X[i])
            if self.augment_temp:
                video = down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = random_skip(video, self.X_skipped_idxs[i])
                video = torch.stack(video)

            X_batch.append(video)

        X_batch = torch.stack(X_batch).unsqueeze(1)

        return X_batch
