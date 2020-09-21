import shutil
import cv2
import torch
import pickle
import numpy as np
import pandas as pd

import sys

sys.path.append("..")
from config import *
from models import get_end2end_model
from utils import ProgressPrinter, get_video_path, get_split_df
from processing_tools import get_tensor_video, get_images, preprocess_3d
from vocab import Vocab, force_alignment


def pad_images(images, stride):
    p = stride // 2
    pad_image = 255 * np.ones(images[0].shape) * np.array([0.406, 0.485, 0.456])
    pad_image = pad_image.astype(np.uint8)
    images = p * [pad_image] + images + p * [pad_image]
    return images


def get_gloss_paths(images, gloss_idx, stride, mode):
    gloss_paths = []
    gloss_lens = []
    shape = images[0].shape[:2][::-1]
    four_cc = cv2.VideoWriter_fourcc(*"mp4v")
    if mode == "3D":  images = pad_images(images, stride)

    s = 0
    while s < len(images):
        e = min(len(images), s + 2 * stride)

        if e - s > stride or (mode == "2D" and e - s == stride):
            gloss_video_path = os.path.join(GR_VIDEOS_DIR, str(gloss_idx) + ".mp4")
            rm_dir = os.path.join(GR_VIDEOS_DIR, str(gloss_idx))
            if os.path.exists(rm_dir):
                shutil.rmtree(rm_dir)

            gloss_images = images[s:e]

            if os.path.exists(gloss_video_path):
                os.remove(gloss_video_path)

            if not os.path.exists(GR_VIDEOS_DIR):
                os.makedirs(GR_VIDEOS_DIR)

            out_cap = cv2.VideoWriter(gloss_video_path, four_cc, 25.0, shape)
            for image in gloss_images:
                out_cap.write(image)
            out_cap.release()

            gloss_paths.append(gloss_video_path)
            gloss_lens.append(len(gloss_images))

            gloss_idx += 1

        s += stride

    return gloss_paths, gloss_lens


def shuffle_and_save_dataset(X, X_lens, Y):
    idxs = list(range(len(X)))
    np.random.shuffle(idxs)
    data = {'X': X, 'X_lens': X_lens, 'Y': Y, "idxs": idxs}

    prefix_dir = os.path.join(GR_DATASET_DIR, "VARS")
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    data_path = os.path.join(prefix_dir, "data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def get_decoded_prediction(model, tensor_video, gt):
    with torch.no_grad():
        pred = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1).cpu().numpy()
        pred = force_alignment(pred, gt)
        glosses = []
        for i in range(len(pred)):
            gloss = pred[i]
            glosses.append(gloss)
        return glosses


def generate_gloss_dataset(vocab, stf_type=STF_TYPE, use_feat=USE_ST_FEAT):
    print("Generation of the Gloss-Recognition Dataset")
    model, loaded = get_end2end_model(vocab, True, stf_type, use_feat)

    mode = "3D" if stf_type else "2D"

    if not loaded:
        print("STF or SEQ2SEQ model doesn't exist")
        exit(0)

    model.eval()

    temp_stride = 4

    rerun_out_dir = os.path.join(GR_DATASET_DIR, "STF_RERUN")
    rerun_out_path = os.path.join(rerun_out_dir, STF_MODEL + ".bin")

    stf_rerun = use_feat and os.path.exists(rerun_out_path)

    if stf_rerun:
        with open(rerun_out_path, 'rb') as f:
            feats_rerun_data = pickle.load(f)
    else:
        feats_rerun_data = {"frame_n": [], "gloss_paths": [], "gloss_lens": []}

    df = get_split_df("train")
    Y = []
    X = []
    X_lens = []

    pp = ProgressPrinter(df.shape[0], 5)
    cur_n_gloss = 0
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        video_path, feat_path = get_video_path(row, "train")

        if stf_rerun:
            frame_n = feats_rerun_data["frame_n"][idx]

            if frame_n < temp_stride:
                pp.omit()
                continue

            gloss_paths = feats_rerun_data["gloss_paths"][idx]
            gloss_lens = feats_rerun_data["gloss_lens"][idx]

            with torch.no_grad():
                tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)

        else:
            images = get_images(video_path)
            frame_n = len(images)
            feats_rerun_data["frame_n"].append(frame_n)

            if frame_n < temp_stride:
                pp.omit()
                feats_rerun_data["gloss_paths"].append("")
                feats_rerun_data["gloss_lens"].append(0)
                continue

            gloss_paths, gloss_lens = get_gloss_paths(images, cur_n_gloss, temp_stride, mode)
            feats_rerun_data["gloss_paths"].append(gloss_paths)
            feats_rerun_data["gloss_lens"].append(gloss_lens)

            with torch.no_grad():
                if use_feat:
                    tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
                else:
                    tensor_video = get_tensor_video(images, preprocess_3d, mode).unsqueeze(0).to(DEVICE)

        X += gloss_paths
        X_lens += gloss_lens
        Y += get_decoded_prediction(model, tensor_video, vocab.encode(row.annotation))

        assert (len(Y) == len(X) == len(X_lens))

        cur_n_gloss = len(X)
        if SHOW_PROGRESS:
            pp.show(idx)

    shuffle_and_save_dataset(X, X_lens, Y)
    if use_feat and not stf_rerun:
        if not os.path.exists(rerun_out_dir): os.makedirs(rerun_out_dir)
        with(open(rerun_out_path, 'wb')) as f:
            pickle.dump(feats_rerun_data, f)

    if SHOW_PROGRESS:
        pp.end()


if __name__ == "__main__":
    vocab = Vocab()
    generate_gloss_dataset(vocab)
