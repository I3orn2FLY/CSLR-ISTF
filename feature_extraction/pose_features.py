import torch
import sys
import numpy as np
from pose import PoseEstimator
import os

sys.path.append(".." + os.sep)
from utils import *
from config import *


def generate_openpose_features_split(pose_estimator, split):
    with torch.no_grad():
        df = get_split_df(split)
        print(SOURCE, "Feature extraction:", IMG_FEAT_MODEL, split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 1)
        for idx in range(L):
            row = df.iloc[idx]
            if SOURCE == "PH":
                video_dir = os.sep.join([VIDEOS_DIR, split, row.folder])
                feat_file = os.sep.join([VIDEO_FEAT_DIR, split, row.folder]).replace("/*.png", ".npy")
            else:
                video_dir = os.path.join(VIDEOS_DIR, row.video)
                feat_file = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".npy")

            if os.path.exists(feat_file):
                pp.omit()
                continue

            feat_dir = os.path.split(feat_file)[0]

            if SOURCE == "PH":
                video = list(glob.glob(video_dir))
                video.sort()
            else:
                video = video_dir

            feats = pose_estimator.estimate_video_pose(video)

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)
            np.save(feat_file, feats)

            pp.show(idx)

        print()


def generate_openpose_features():
    if IMG_FEAT_MODEL not in ["pose"]:
        print("Incorrect feature extraction model:", IMG_FEAT_MODEL)
        exit(0)

    pose_estimator = PoseEstimator()
    generate_openpose_features_split(pose_estimator, "train")
    generate_openpose_features_split(pose_estimator, "dev")
    generate_openpose_features_split(pose_estimator, "test")


if __name__ == "__main__":
    generate_openpose_features()
    # generate_cnn_features()
    # generate_numpy_videos(source=HANDS_DIR, dest=HANDS_NP_IMGS_DIR, side=HAND_SIZE)

    # generate_gloss_dataset(with_blank=False)
