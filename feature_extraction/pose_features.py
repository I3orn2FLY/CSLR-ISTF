import torch
import sys
import numpy as np
from utils import *
from pose import PoseEstimator

sys.path.append(os.sep.join(["..", "*"]))
from config import *


def generate_openpose_features_split(pose_estimator, split):
    if SOURCE == "PH":
        pose_feat_dir = os.sep.join([PH_DIR, "features", "pose"])
    else:
        pose_feat_dir = os.sep.join([KRSL_DIR, "features", "pose"])

        if not os.path.exists(pose_feat_dir):
            os.makedirs(pose_feat_dir)

    with torch.no_grad():
        df = get_pheonix_df(split)
        print("Feature extraction:", split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 1)
        for idx in range(L):
            row = df.iloc[idx]
            img_dir = os.sep.join([PH_IMAGES_DIR, split, row.folder])
            feat_dir = os.sep.join([pose_feat_dir, split, row.folder])
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





if __name__ == "__main__":
    generate_openpose_features()
    # generate_cnn_features()
    # generate_numpy_videos(source=HANDS_DIR, dest=HANDS_NP_IMGS_DIR, side=HAND_SIZE)

    # generate_gloss_dataset(with_blank=False)
