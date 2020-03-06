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

    def estimate_video_pose(self, image_files):
        video_pose = []

        for image_file in image_files:
            frame = cv2.imread(image_file)
            datum = self.estimate_pose(frame)

            body = datum.poseKeypoints
            try:
                num_people = body.shape[0]
            except:
                continue
            if num_people > 1 or num_people < 1: continue
            body = body.squeeze()
            face = datum.faceKeypoints.squeeze()
            hleft = datum.handKeypoints[0].squeeze()
            hright = datum.handKeypoints[1].squeeze()
            pose_data = np.vstack((face, body, hleft, hright)).reshape(-1)
            video_pose.append(pose_data)


        video_pose = np.array(video_pose)

        return video_pose
