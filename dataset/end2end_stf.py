import torch

from dataset.end2end_base import End2EndDataset, random_skip, down_sample

from config import *
from utils import Vocab
from common import get_video_path


class End2EndSTFDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        super(End2EndSTFDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_feat(self, row, glosses=None):
        video_path, feat_path = get_video_path(row, self.split)

        if not os.path.exists(feat_path):
            return None, None, None

        feat = torch.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) or len(feat.shape) < 2:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, idx):
        batch_idxs = self.batches[idx]
        X_batch = []
        for i in batch_idxs:

            video = torch.load(self.X[i])
            if self.augment_temp:
                video = down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = random_skip(video, self.X_skipped_idxs[i])
                video = torch.stack(video)

            X_batch.append(video)

        X_batch = torch.stack(X_batch)

        return X_batch

    def _get_aug_diff(self, L, out_seq_len):
        return L - out_seq_len


if __name__ == "__main__":
    vocab = Vocab()
    dataset = End2EndSTFDataset(vocab, "train", 32, True, True)

    dataset.start_epoch()

    X_batch, Y_batch, Y_lens = dataset.get_batch(0)

    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
