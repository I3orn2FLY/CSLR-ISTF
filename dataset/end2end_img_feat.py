import torch
import numpy as np

from dataset.end2end_base import End2EndDataset, random_skip, down_sample

from config import *
from utils import get_video_path
from vocab import Vocab


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


class End2EndImgFeatDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        super(End2EndImgFeatDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_ffm(self):
        return os.path.join("IMG_FEAT", STF_MODEL + "_" + str(IMG_FEAT_SIZE))

    def _get_feat(self, row, glosses=None):
        ext = ".npy" if STF_MODEL.startswith("pose") else ".pt"
        video_path, feat_path = get_video_path(row, self.split, feat_ext=ext, stf_feat=False)

        if not os.path.exists(feat_path):
            return None, None, None

        feat = np.load(feat_path) if STF_MODEL.startswith("pose") else torch.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) or len(feat.shape) < 2:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, idx):
        batch_idxs = self.batches[idx]
        X_batch = []
        for i in batch_idxs:
            if STF_MODEL.startswith("pose"):
                video = np.load(self.X[i])
                video = process_video_pose(video, augment_frame=self.augment_frame)
            else:
                video = torch.load(self.X[i])
            if self.augment_temp:
                video = down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = random_skip(video, self.X_skipped_idxs[i])
                video = np.stack(video) if STF_MODEL.startswith("pose") else torch.stack(video)

            X_batch.append(video)

        if STF_MODEL.startswith("pose"):
            X_batch = torch.from_numpy(np.stack(X_batch).astype(np.float32)).unsqueeze(1)
        else:
            X_batch = torch.stack(X_batch)

        return X_batch

    def _get_aug_diff(self, L, out_seq_len):
        return L - out_seq_len


if __name__ == "__main__":
    vocab = Vocab()
    dataset = End2EndImgFeatDataset(vocab, "train", 32, True, True)

    dataset.start_epoch()

    X_batch, Y_batch, Y_lens = dataset.get_batch(0)

    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
