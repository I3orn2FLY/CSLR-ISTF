import shutil
import cv2
import torch
import numpy as np
import pandas as pd
from config import *
from models import get_end2end_model
from utils import ProgressPrinter, get_video_path, get_split_df
from processing_tools import get_tensor_video, get_images, preprocess_3d
from vocab import Vocab


# change for 2d data

def min_dist_transform(hypo, gt):
    dp = [[None for _ in range(len(gt) + 1)] for _ in range(len(hypo) + 1)]

    def ed(i, j):
        if dp[i + 1][j + 1] is not None:
            return dp[i + 1][j + 1]

        if i == -1 and j == -1:
            dp[0][0] = []
            return []

        if i == -1:
            # insert
            dp[0][j] = ed(-1, j - 1)
            dp[0][j + 1] = ["ins_0_" + str(gt[j])] + dp[i + 1][j]

            return dp[0][j + 1]

        if j == -1:
            # delete
            dp[i][0] = ed(i - 1, -1)

            dp[i + 1][0] = ["del_" + str(i)] + dp[i][0]

            return dp[i + 1][0]

        if hypo[i] == gt[j]:
            dp[i + 1][j + 1] = ed(i - 1, j - 1)
            return dp[i + 1][j + 1]
        else:
            # del
            dp[i][j + 1] = ed(i - 1, j)
            opt = dp[i][j + 1]
            des = "del_" + str(i)

            # ins
            dp[i + 1][j] = ed(i, j - 1)
            if len(dp[i + 1][j]) < len(opt):
                opt = dp[i + 1][j]
                des = "ins_" + str(i + 1) + "_" + str(gt[j])

            # rep
            dp[i][j] = ed(i - 1, j - 1)

            if len(dp[i][j]) < len(opt):
                opt = dp[i][j]
                des = "rep_" + str(i) + "_" + str(gt[j])

            dp[i + 1][j + 1] = [des] + opt

            return dp[i + 1][j + 1]

    return ed(len(hypo) - 1, len(gt) - 1)


def force_alignment(pred, gt):
    hypo = []
    hypo_idxs = []
    s = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            s = i + 1
            continue

        if i < len(pred) - 1 and pred[i] == pred[i + 1]:
            continue

        hypo.append(pred[i])
        hypo_idxs.append((s, i + 1))

        s = i + 1

    transform = min_dist_transform(hypo, gt)

    for oper in transform:
        oper = oper.split("_")
        des = oper[0]
        idx = int(oper[1])
        val = int(oper[-1])

        if des == "del":
            del hypo[idx]
            del hypo_idxs[idx]
        elif des == "ins":
            if idx > 0 and idx < len(hypo) - 1:
                s1, e1 = hypo_idxs[idx - 1]
                s2, e2 = hypo_idxs[idx]
                l1 = hypo[idx - 1]
                l2 = hypo[idx]
            elif idx == 0:
                s1, e1 = 0, 0
                s2, e2 = hypo_idxs[0]
                l1 = 0
                l2 = hypo[0]
            else:
                s1, e1 = hypo_idxs[-1]
                s2, e2 = len(pred), len(pred)
                l1 = hypo[-1]
                l2 = 0

            if l1 == val == l2:
                continue

            if s2 - e1 > 1:
                s = e1
                e = s2
                if l1 == val:
                    s += 1
                elif l2 == val:
                    e -= 1

                if e - s > 2:
                    s = s + (e - s - 2) // 2
                    e = s + 2

                hypo_idxs.insert(idx, (s, e))
                hypo.insert(idx, val)

        else:
            hypo[idx] = val

    pred = [0] * len(pred)

    for label, (s, e) in zip(hypo, hypo_idxs):
        for i in range(s, e):
            pred[i] = label

    return pred


def get_gloss_paths(images, pad_image, gloss_idx, stride, save=True):
    gloss_paths = []

    s = 0
    p = stride // 2

    images = p * [pad_image] + images + p * [pad_image]
    while s < len(images):
        e = min(len(images), s + 2 * stride)
        if e - s > stride:
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
    if not STF_MODEL.startswith("resnet{2+1}d") or stf_type != 1:
        print("Incorrect feature extraction model:", STF_MODEL, STF_TYPE)
        exit(0)

    print("Genearation of the Gloss-Recognition Dataset")
    model, loaded = get_end2end_model(vocab, True, stf_type, use_feat)

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
            gloss_paths += get_gloss_paths(images, pad_image, gloss_idx, temp_stride)
            if use_feat:
                tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
            else:
                tensor_video = get_tensor_video(images, preprocess_3d, "3D").unsqueeze(0).to(DEVICE)

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
