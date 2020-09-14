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


def get_gloss_paths(images, pad_image, gloss_idx, stride, mode, resave=True):
    gloss_paths = []

    s = 0
    p = stride // 2

    if mode == "3D":
        images = p * [pad_image] + images + p * [pad_image]
    while s < len(images):
        e = min(len(images), s + 2 * stride)

        if e - s > stride or (mode == "2D" and e - s == stride):
            gloss_video_dir = os.path.join(GR_VIDEOS_DIR, str(gloss_idx))

            if resave:
                gloss_images = images[s:e]
                if os.path.exists(gloss_video_dir):
                    shutil.rmtree(gloss_video_dir)

                if not os.path.exists(gloss_video_dir):
                    os.makedirs(gloss_video_dir)

                for idx, image in enumerate(gloss_images):
                    if not os.path.exists(os.path.join(gloss_video_dir, str(idx) + ".jpg")):
                        cv2.imwrite(os.path.join(gloss_video_dir, str(idx) + ".jpg"), image)

            gloss_paths.append(os.path.join(str(gloss_idx), "*.jpg"))

            gloss_idx += 1

        s += stride

    return gloss_paths


def shuffle_and_save_csv(out_all_paths, Y):
    Y_gloss = [vocab.idx2gloss[i] for i in Y]
    df = pd.DataFrame({"folder": out_all_paths, "gloss": Y_gloss, "gloss_idx": Y})

    L = df.shape[0]
    idxs = list(range(L))
    np.random.shuffle(idxs)
    df_train = df.iloc[idxs[:int(0.9 * L)]]
    df_val = df.iloc[idxs[int(0.9 * L):]]

    if not os.path.exists(GR_ANNO_DIR):
        os.makedirs(GR_ANNO_DIR)

    df_train.to_csv(os.path.join(GR_ANNO_DIR, "gloss_train.csv"), index=None)
    df_val.to_csv(os.path.join(GR_ANNO_DIR, "gloss_val.csv"), index=None)


def get_decoded_prediction(model, tensor_video, gt):
    pred = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1).cpu().numpy()
    pred = force_alignment(pred, gt)
    glosses = []
    for i in range(len(pred)):
        gloss = pred[i]
        glosses.append(gloss)
    return glosses


def generate_gloss_dataset_with_only_feats(model, mode):
    gr_out_dir = os.path.join(GR_ANNO_DIR, "GR_OUT")
    gr_out_path = os.path.join(gr_out_dir, mode + ".bin")
    if not os.path.exists(gr_out_path): return False

    print("Generating based on existing Gloss Dataset")
    with(open(gr_out_path, 'rb')) as f:
        gr_out = pickle.load(f)

    in_feats_paths = gr_out["in_feats_paths"]
    out_glosses_paths = gr_out["out_glosses_paths"]
    gts = gr_out["gts"]

    Y = []
    out_all_paths = []
    for feat_path, annotation, out_video_paths in zip(in_feats_paths, gts, out_glosses_paths):
        if not os.path.exists(feat_path):
            print("ERROR")
            exit(0)

        tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
        Y += get_decoded_prediction(model, tensor_video, annotation)
        out_all_paths += out_video_paths
        assert (len(Y) == len(out_all_paths))

    shuffle_and_save_csv(out_all_paths, Y)


def generate_gloss_dataset(vocab, stf_type=STF_TYPE, use_feat=USE_ST_FEAT, resave=True):
    print("Generation of the Gloss-Recognition Dataset")
    model, loaded = get_end2end_model(vocab, True, stf_type, use_feat)

    mode = "3D" if stf_type else "2D"

    if not loaded:
        print("STF or SEQ2SEQ model doesn't exist")
        exit(0)

    model.eval()

    if use_feat and generate_gloss_dataset_with_only_feats(model, mode): exit(0)

    pad_image = 255 * np.ones((260, 210, 3)) * np.array([0.406, 0.485, 0.456])

    pad_image = pad_image.astype(np.uint8)

    temp_stride = 4

    df = get_split_df("train")
    Y = []
    out_all_paths = []

    gts = []
    out_glosses_paths = []
    in_feats_paths = []
    with torch.no_grad():

        pp = ProgressPrinter(df.shape[0], 5)
        cur_n_gloss = 0

        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            video_path, feat_path = get_video_path(row, "train")
            images = get_images(video_path)
            if len(images) < 4:
                pp.omit()
                continue

            out_video_paths = get_gloss_paths(images, pad_image, cur_n_gloss, temp_stride, mode, resave)
            out_all_paths += out_video_paths

            if use_feat:
                tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
            else:
                tensor_video = get_tensor_video(images, preprocess_3d, mode).unsqueeze(0).to(DEVICE)

            gt = vocab.encode(row.annotation)
            Y += get_decoded_prediction(model, tensor_video, gt)

            gts.append(gt)
            in_feats_paths.append(feat_path)
            out_glosses_paths.append(out_video_paths)

            assert (len(Y) == len(out_all_paths))

            cur_n_gloss = len(Y)

            if SHOW_PROGRESS:
                pp.show(idx)

        if SHOW_PROGRESS:
            pp.end()

    shuffle_and_save_csv(out_all_paths, Y)
    if use_feat:
        gr_out_dir = os.path.join(GR_ANNO_DIR, "GR_OUT")
        gr_out_path = os.path.join(gr_out_dir, mode + ".bin")
        gr_out = {"in_feats_paths": in_feats_paths,
                  "out_glosses_paths": out_glosses_paths,
                  "gts": gts}

        if not os.path.exists(gr_out_dir): os.makedirs(gr_out_dir)
        with(open(gr_out_path, 'wb')) as f:
            pickle.dump(gr_out, f)


if __name__ == "__main__":
    vocab = Vocab()
    generate_gloss_dataset(vocab)
