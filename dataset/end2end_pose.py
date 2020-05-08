import torch
import numpy as np

from dataset.end2end_base import End2EndDataset, random_skip, down_sample
from utils import Vocab

from config import *


# TODO implement loading pose data and generate stf feats from 1d cnns maybe new code for stf_feats is needed

def process_video_pose(video_pose, augment_frame=True):
    video_pose = video_pose.reshape(-1, 137, 3)
    idxs = []
    video_pose = video_pose[:, :, :2]

    noise = []
    if POSE_FACE:
        idxs += list(range(70))
        noise.append(POSE_AUG_NOISE_HANDFACE - 2 * POSE_AUG_NOISE_HANDFACE * np.random.rand(len(video_pose), 70, 2))

    if POSE_BODY:
        idxs += list(range(70, 70 + 8)) + list(range(70 + 15, 70 + 19))
        noise.append(POSE_AUG_NOISE_BODY - 2 * POSE_AUG_NOISE_BODY * np.random.rand(len(video_pose), 12, 2))

    if POSE_HANDS:
        idxs += list(range(95, 137))
        noise.append(POSE_AUG_NOISE_HANDFACE - 2 * POSE_AUG_NOISE_HANDFACE * np.random.rand(len(video_pose), 42, 2))

    video_pose = video_pose[:, idxs]

    if augment_frame:
        noise = np.concatenate(noise, axis=1)
        offset = POSE_AUG_OFFSET - 2 * POSE_AUG_OFFSET * np.random.rand(2)
        video_pose += noise + offset

    return video_pose.reshape(len(video_pose), -1)


class End2EndPoseDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if (not STF_MODEL.startswith("pose")) or SRC_MODE == "HAND" or (not USE_STF_FEAT):
            print("Incorrect params => ", STF_MODEL, SRC_MODE, "USE_FEAT:", USE_STF_FEAT)
            exit(0)
        super(End2EndPoseDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_feat(self, row, glosses=None):
        if SOURCE == "PH":
            feat_path = os.sep.join([STF_FEAT_DIR, self.split, row.folder.replace("/1/*.png", ".npy")])
        elif SOURCE == "KRSL":
            feat_path = os.path.join(STF_FEAT_DIR, row.video).replace(".mp4", ".npy")
        else:
            return None, None, None

        if not os.path.exists(feat_path):
            return None, None, None

        feat = np.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) * 4:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = np.load(self.X[i])
            video = process_video_pose(video, augment_frame=self.augment_frame)
            if self.augment_temp:
                video = down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = random_skip(video, self.X_skipped_idxs[i])
                video = np.stack(video)

            X_batch.append(video)

        X_batch = torch.from_numpy(np.stack(X_batch).astype(np.float32)).unsqueeze(1)

        return X_batch


if __name__ == "__main__":
    vocab = Vocab()
    dataset = End2EndPoseDataset(vocab, "train", 32, True, True)

    dataset.start_epoch()

    X_batch, Y_batch, Y_lens = dataset.get_batch(0)

    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
