import sys
import numpy as np
import cv2
import os

sys.path.append(".." + os.sep)
from utils import *
from config import *


# TODO change the code for new dataset adaptation

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


def generate_numpy_videos(source, dest, side):
    generate_numpy_videos_split("train", source, dest, side)
    generate_numpy_videos_split("dev", source, dest, side)
    generate_numpy_videos_split("test", source, dest, side)
    generate_numpy_image_mean(dest, side)
    generate_numpy_image_std(dest, side)


if __name__ == "__main__":
    generate_numpy_videos(source=HANDS_DIR, dest=HANDS_NP_IMGS_DIR, side=HAND_SIZE)
