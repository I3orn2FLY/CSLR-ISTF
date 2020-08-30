import shutil
import cv2
import torch
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
from config import *
from models import get_end2end_model
from utils import ProgressPrinter, get_video_path, get_split_df
from processing_tools import get_tensor_video, get_images, preprocess_3d
from vocab import Vocab, force_alignment


def get_gloss_paths(images, pad_image, gloss_idx, stride, mode, save=True):
    gloss_paths = []

    s = 0
    p = stride // 2

    if mode == "3D":
        images = p * [pad_image] + images + p * [pad_image]
    while s < len(images):
        e = min(len(images), s + 2 * stride)
        # temporary fix about 2D mode and 3D returns different output lengths

        if e - s > stride or (mode == "2D" and e - s == stride):
            gloss_video_dir = os.path.join(GR_VIDEOS_DIR, str(gloss_idx))

            if save:
                gloss_images = images[s:e]
                if os.path.exists(gloss_video_dir):
                    shutil.rmtree(gloss_video_dir)

                os.makedirs(gloss_video_dir)

                for idx, image in enumerate(gloss_images):
                    cv2.imwrite(os.path.join(gloss_video_dir, str(idx) + ".jpg"), image)

            gloss_paths.append(os.path.join(str(gloss_idx), "*.jpg"))

            gloss_idx += 1

        s += stride

    return gloss_paths


def generate_gloss_dataset(vocab, stf_type=STF_TYPE, use_feat=USE_ST_FEAT):
    print("Generation of the Gloss-Recognition Dataset")
    model, loaded = get_end2end_model(vocab, True, stf_type, use_feat)
    if stf_type == 0:
        mode = "2D"
    else:
        mode = "3D"
    if not loaded:
        print("STF or SEQ2SEQ model doesn't exist")
        exit(0)

    model.eval()

    pad_image = 255 * np.ones((260, 210, 3)) * np.array([0.406, 0.485, 0.456])

    pad_image = pad_image.astype(np.uint8)

    temp_stride = 4
    df = get_split_df("train")
    Y = []
    gloss_paths = []
    with torch.no_grad():

        pp = ProgressPrinter(df.shape[0], 5)
        gloss_idx = 0

        for idx in range(df.shape[0]):

            row = df.iloc[idx]

            video_path, feat_path = get_video_path(row, "train")

            images = get_images(video_path)
            if len(images) < 4:
                continue

            gloss_paths += get_gloss_paths(images, pad_image, gloss_idx, temp_stride, mode)
            if use_feat:
                tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
            else:
                tensor_video = get_tensor_video(images, preprocess_3d, mode).unsqueeze(0).to(DEVICE)

            pred = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1).cpu().numpy()

            gt = vocab.encode(row.annotation)
            pred = force_alignment(pred, gt)

            for i in range(len(pred)):
                gloss = pred[i]
                Y.append(gloss)

            assert (len(Y) == len(gloss_paths))

            gloss_idx = len(Y)
            if SHOW_PROGRESS:
                pp.show(idx)

        if SHOW_PROGRESS:
            pp.end()

    Y_gloss = [vocab.idx2gloss[i] for i in Y]

    df = pd.DataFrame({"folder": gloss_paths, "gloss": Y_gloss, "gloss_idx": Y})

    L = df.shape[0]
    idxs = list(range(L))
    np.random.shuffle(idxs)
    df_train = df.iloc[idxs[:int(0.9 * L)]]
    df_val = df.iloc[idxs[int(0.9 * L):]]

    if not os.path.exists(GR_ANNO_DIR):
        os.makedirs(GR_ANNO_DIR)

    df_train.to_csv(os.path.join(GR_ANNO_DIR, "gloss_train.csv"), index=None)
    df_val.to_csv(os.path.join(GR_ANNO_DIR, "gloss_val.csv"), index=None)


if __name__ == "__main__":
    vocab = Vocab()
    generate_gloss_dataset(vocab)
