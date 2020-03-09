import torch
import sys
import glob
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
from models import FrameFeatModel, SLR
from utils import ProgressPrinter, Vocab
from dataset import read_pheonix_cnn_feats, get_pheonix_df
from pose import PoseEstimator

sys.path.append(os.sep.join(["..", "*"]))
from config import *


def generate_openpose_features_split(pose_estimator, split):
    with torch.no_grad():
        df = get_pheonix_df(split)
        print("Feature extraction:", split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 1)
        for idx in range(L):
            row = df.iloc[idx]
            img_dir = os.sep.join([IMAGES_DIR, split, row.folder])
            feat_dir = os.sep.join([POSE_FEAT_DIR, split, row.folder])
            feat_file = feat_dir.replace("/*.png", "")

            if os.path.exists(feat_file + ".npy"):
                pp.omit()
                continue

            feat_dir = os.path.split(feat_file)[0]

            image_files = list(glob.glob(img_dir))
            image_files.sort()

            feats = pose_estimator.estimate_video_pose(image_files)

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)
            np.save(feat_file, feats)

            pp.show(idx)

        print()


def generate_openpose_features():
    pose_estimator = PoseEstimator()

    generate_openpose_features_split(pose_estimator, "train")
    generate_openpose_features_split(pose_estimator, "dev")
    generate_openpose_features_split(pose_estimator, "test")


def generate_cnn_features_split(model, device, preprocess, split):
    with torch.no_grad():
        df = get_pheonix_df(split)
        print("Feature extraction:", split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 10)
        for idx in range(L):
            row = df.iloc[idx]
            img_dir = os.sep.join([IMAGES_DIR, split, row.folder])
            feat_dir = os.sep.join([VIDEO_FEAT_DIR, split, row.folder])
            feat_file = feat_dir.replace("/*.png", "")

            if os.path.exists(feat_file + ".npy"):
                continue

            feat_dir = os.path.split(feat_file)[0]

            image_files = list(glob.glob(img_dir))
            image_files.sort()

            images = [Image.open(img_file) for img_file in image_files]
            inp = torch.stack([preprocess(image) for image in images])
            inp = inp.to(device)
            feats = model(inp).cpu().numpy()

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)
            np.save(feat_file, feats)

            pp.show(idx)

        print()


def generate_cnn_features():
    device = DEVICE
    model = FrameFeatModel().to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    generate_cnn_features_split(model, device, preprocess, "train")
    generate_cnn_features_split(model, device, preprocess, "test")
    generate_cnn_features_split(model, device, preprocess, "dev")


def generate_numpy_image_mean(dest, side):
    print("Generating numpy image mean")
    split = "train"

    mean_img_path = os.path.join(VARS_DIR, os.path.split(dest)[1] + "_mean.npy")
    if os.path.exists(mean_img_path):
        return

    n = 0
    mean_img = np.zeros((3, side, side))
    df = get_pheonix_df(split)
    pp = ProgressPrinter(df.shape[0], 25)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        feat_dir = os.sep.join([dest, split, row.folder])
        np_video_file = feat_dir.replace("/*.png", ".npy")
        if not os.path.exists(np_video_file):
            print("ERROR NUMPY FILE DOES NOT EXIST:", np_video_file)
            return

        imgs = np.load(np_video_file)

        mean_img += np.sum(imgs, axis=0)

        n += len(imgs)
        pp.show(idx)

    mean_img /= n
    pp.end()
    np.save(mean_img_path.replace(".npy", ""), mean_img)


def generate_numpy_image_std(dest, side):
    print("Generating numpy image std")
    split = "train"
    mean_img_path = os.path.join(VARS_DIR, os.path.split(dest)[1] + "_mean.npy")
    std_img_path = os.path.join(VARS_DIR, os.path.split(dest)[1] + "_std.npy")
    if not os.path.exists(mean_img_path) or os.path.exists(std_img_path):
        return

    mean_img = np.load(mean_img_path)
    n = 0
    std_img = np.zeros((3, side, side))
    df = get_pheonix_df(split)
    pp = ProgressPrinter(df.shape[0], 25)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        feat_dir = os.sep.join([dest, split, row.folder])
        np_video_file = feat_dir.replace("/*.png", ".npy")
        if not os.path.exists(np_video_file):
            print("ERROR NUMPY FILE DOES NOT EXIST:", np_video_file)
            return

        imgs = np.load(np_video_file)

        std_img += np.sum(np.square(imgs - mean_img), axis=0)

        n += len(imgs)
        pp.show(idx)

    std_img /= n

    std_img = np.sqrt(std_img)

    np.save(std_img_path.replace(".npy", ""), std_img)
    pp.end()


def generate_numpy_videos_split(split, source, dest, side):
    print("Generating numpy video", split, "split")
    df = get_pheonix_df(split)
    pp = ProgressPrinter(df.shape[0], 2)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        img_dir = os.sep.join([source, split, row.folder])
        feat_dir = os.sep.join([dest, split, row.folder])
        np_video_file = feat_dir.replace("/*.png", "")
        if os.path.exists(np_video_file + ".npy"):
            pp.omit()
            continue

        feat_dir = os.path.split(np_video_file)[0]

        image_files = list(glob.glob(img_dir))
        image_files.sort()
        imgs = np.array([cv2.resize(cv2.imread(img_file), (side, side)) for img_file in image_files])
        imgs = imgs.transpose([0, 3, 1, 2])
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        np.save(np_video_file, imgs)

        pp.show(idx)

    pp.end()


def generate_numpy_videos(source, dest, side):
    generate_numpy_videos_split("train", source, dest, side)
    generate_numpy_videos_split("dev", source, dest, side)
    generate_numpy_videos_split("test", source, dest, side)
    generate_numpy_image_mean(dest, side)
    generate_numpy_image_std(dest, side)


# def generate_gloss_dataset(with_blank=True):
#     vocab = Vocab(source="pheonix")
#     device = DEVICE
#     model = SLR(rnn_hidden=512, vocab_size=vocab.size).to(device)
#     model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr.pt"])))
#     model.eval()
#
#     X_tr, y_tr = read_pheonix_cnn_feats("train", vocab, save=True)
#     X_batches, y_batches = split_batches(X_tr, y_tr, 16, shuffle=False, target_format=2)
#     stride = 4
#     X = []
#     y = []
#     with torch.no_grad():
#         pp = ProgressPrinter(len(y_batches), 10)
#         for idx in range(len(X_batches)):
#             X_batch = X_batches[idx]
#             inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
#             preds = model(inp).log_softmax(dim=2).permute(1, 0, 2).cpu().numpy().argmax(axis=2)
#             for i in range(preds.shape[0]):
#                 for j in range(len(preds[i])):
#                     feat = X_batch[i][j * stride: (j + 1) * stride]
#                     gloss = preds[i][j]
#                     if not with_blank and gloss == 0:
#                         continue
#                     X.append(feat)
#                     y.append(gloss)
#
#             pp.show(idx)
#
#     print()
#     assert len(X) == len(y), "ASD"
#
#     X = np.array(X)
#     y = np.array(y).astype(np.int32)
#     idxs = list(range(len(y)))
#     np.random.shuffle(idxs)
#     tr = int(0.9 * len(y))
#
#     X_tr = X[:tr]
#     y_tr = y[:tr]
#
#     X_dev = X[tr:]
#     y_dev = y[tr:]
#
#     X_path = os.sep.join([VARS_DIR, "X_gloss_"])
#     y_path = os.sep.join([VARS_DIR, "y_gloss_"])
#     if not with_blank:
#         X_path += "no_blank_"
#         y_path += "no_blank_"
#
#     np.save(X_path + "train", X_tr)
#     np.save(y_path + "train", y_tr)
#     np.save(X_path + "dev", X_dev)
#     np.save(y_path + "dev", y_dev)
#
#     print(X_tr.shape, y_tr.shape)
#     print(X_dev.shape, y_dev.shape)


if __name__ == "__main__":
    generate_openpose_features()
    # generate_cnn_features()
    # generate_numpy_videos(source=HANDS_DIR, dest=HANDS_NP_IMGS_DIR, side=HAND_SIZE)

    # generate_gloss_dataset(with_blank=False)
