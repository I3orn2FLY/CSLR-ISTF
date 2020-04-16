import sys
import cv2
import numpy as np
import os

sys.path.append(os.path.join("..", "*"))
from config import *

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
