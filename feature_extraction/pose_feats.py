import torch
import sys
import numpy as np

from utils import *
from config import *
from common import get_video_path

sys.path.append(os.path.join(OPENPOSE_FOLDER, "build/python"))
from openpose import pyopenpose as op


class PoseEstimator():
    def __init__(self, hand=True, face=True):
        params = dict()
        params["model_folder"] = os.path.join(OPENPOSE_FOLDER, "models")
        params["face"] = hand
        params["hand"] = face
        # for scaling
        params["keypoint_scale"] = 3
        params["num_gpu"] = 1
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate_pose(self, imageToProcess):
        datum = op.Datum()
        if not isinstance(imageToProcess, np.ndarray):
            return None
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])

        return datum

    def estimate_video_pose(self, video):
        video_pose = []

        if isinstance(video, list):
            for image_file in video:
                frame = cv2.imread(image_file)

                pose = self.estimate_image_pose(frame)
                if isinstance(pose, np.ndarray):
                    video_pose.append(pose)


        elif isinstance(video, str):
            cap = cv2.VideoCapture(video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pose = self.estimate_image_pose(frame)
                if isinstance(pose, np.ndarray):
                    video_pose.append(pose)

            cap.release()

        video_pose = np.array(video_pose)

        return video_pose

    def estimate_image_pose(self, frame):
        datum = self.estimate_pose(frame)

        body = datum.poseKeypoints
        try:
            num_people = body.shape[0]
        except:
            return None

        idx = 0
        if num_people < 1:
            return None
        elif num_people > 1:
            confs = np.sum(body, axis=1)[:, 2]
            idx = np.argmax(confs)

        body = body[idx]
        face = datum.faceKeypoints[idx]
        hleft = datum.handKeypoints[0][idx]
        hright = datum.handKeypoints[1][idx]
        return np.vstack((face, body, hleft, hright)).reshape(-1)


def generate_openpose_features_split(pose_estimator, split):
    with torch.no_grad():
        df = get_split_df(split)
        print(SOURCE, "Feature extraction:", STF_MODEL, split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 1)
        for idx in range(L):
            row = df.iloc[idx]
            video_dir, feat_path = get_video_path(row, split, feat_ext=".npy")

            if os.path.exists(feat_path):
                pp.omit()
                continue

            feat_dir = os.path.split(feat_path)[0]

            feats = pose_estimator.estimate_video_pose(video_dir)

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)
            np.save(feat_path, feats)

            if SHOW_PROGRESS:
                pp.show(idx)

        if SHOW_PROGRESS:
            pp.end()

        print()


def generate_openpose_features():
    if STF_MODEL not in ["pose"]:
        print("Incorrect feature extraction model:", STF_MODEL)
        exit(0)

    pose_estimator = PoseEstimator()
    generate_openpose_features_split(pose_estimator, "train")
    generate_openpose_features_split(pose_estimator, "dev")
    generate_openpose_features_split(pose_estimator, "test")


if __name__ == "__main__":
    generate_openpose_features()
