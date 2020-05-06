import pickle
import torch
import numpy as np


from config import *
from utils import get_split_df, ProgressPrinter


def down_sample(video, n):
    video = [video[int(i)] for i in np.linspace(0, len(video) - 1, n)]
    return video


def random_skip(video, skipped_idxs):
    res_video = []

    for i in range(len(video)):
        if skipped_idxs and i == skipped_idxs[0]:
            skipped_idxs.pop(0)
            continue

        res_video.append(video[i])

    return res_video


def crop_video(video):
    cropped_video = []
    for img in video:
        h, w = img.shape[:2]
        y1, x1 = int(0.2 * np.random.rand() * h), int(0.2 * np.random.rand() * h)
        y2, x2 = h - int(0.2 * np.random.rand() * h), w - int(0.2 * np.random.rand() * h)
        img = img[y1:y2, x1:x2]
        cropped_video.append(img)

    return cropped_video


def noise_video(video):
    video = video.astype(np.float32)
    video += 2 - 4 * np.random.rand(*video.shape)

    video = np.maximum(video, 0)
    video = np.minimum(video, 255)

    # video = video.astype(np.uint8)
    return video


class End2EndDataset():
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if split == "train":
            self.augment_temp = augment_temp
            self.augment_frame = augment_frame
        else:
            self.augment_temp = False
            self.augment_frame = False

        self.max_batch_size = max_batch_size

        if SOURCE == "PH" and split == "val":
            split = "dev"

        if SOURCE == "KRSL" and split == "dev":
            split = "val"

        self.split = split
        self.vocab = vocab
        self._build_dataset()

    def _get_feat(self, row, glosses=None):
        raise NotImplementedError

    def _show_progress(self):
        return False

    def _get_ffm(self):
        return IMG_FEAT_MODEL

    def _build_dataset(self):

        dataset_dir = os.sep.join([ENDEND_DATASETS_DIR, self._get_ffm()])

        X_path = os.sep.join([dataset_dir, "X_" + self.split + ".pkl"])
        Y_path = os.sep.join([dataset_dir, "Y_" + self.split + ".pkl"])
        X_lens_path = os.sep.join([dataset_dir, "X_lens_" + self.split + ".pkl"])

        if os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(X_lens_path):
            with open(X_path, 'rb') as f:
                self.X = pickle.load(f)

            with open(Y_path, 'rb') as f:
                self.Y = pickle.load(f)

            with open(X_lens_path, 'rb') as f:
                self.X_lens = pickle.load(f)

            print(self.split[0].upper() + self.split[1:], "dataset loaded")
        else:
            print("Building", self.split, "dataset")
            df = get_split_df(self.split)
            self.X = []
            self.Y = []
            self.X_lens = []

            pp = ProgressPrinter(df.shape[0], 5)
            for idx in range(df.shape[0]):
                row = df.iloc[idx]
                glosses = self.vocab.encode(row.annotation)
                feat_path, feat, feat_len = self._get_feat(row, glosses)
                if feat is None:
                    continue

                self.X.append(feat_path)
                self.Y.append(glosses)
                self.X_lens.append(feat_len)

                if self._show_progress():
                    pp.show(idx)

            if self._show_progress():
                pp.end()

            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            with open(X_path, 'wb') as f:
                pickle.dump(self.X, f)

            with open(Y_path, 'wb') as f:
                pickle.dump(self.Y, f)

            with open(X_lens_path, 'wb') as f:
                pickle.dump(self.X_lens, f)

        self.length = len(self.X)

    def start_epoch(self, shuffle=True):
        self.X_aug_lens, self.X_skipped_idxs = self._get_aug_input_lens()
        len_table = dict()

        for i, length in enumerate(self.X_aug_lens):
            if length in len_table:
                len_table[length].append(i)
            else:
                len_table[length] = [i]

        self.batches = []
        lenghts = list(len_table)

        if shuffle:
            np.random.shuffle(lenghts)

        for l in lenghts:
            idxs = len_table[l]
            if shuffle:
                np.random.shuffle(idxs)
            s = 0
            while (s < len(idxs)):
                e = min(s + self.max_batch_size, len(idxs))

                self.batches.append(idxs[s:e])

                s += self.max_batch_size

        return len(self.batches)

    def get_X_batch(self, batch_idxs):

        raise NotImplementedError

    def get_batch(self, idx):
        batch_idxs = self.batches[idx]
        Y_lens = [len(self.Y[i]) for i in batch_idxs]

        X_batch = self.get_X_batch(batch_idxs)

        max_target_length = max(Y_lens)

        Y_batch = np.zeros((len(batch_idxs), max_target_length), dtype=np.int32)

        for idx, i in enumerate(batch_idxs):
            Y_batch[idx][:len(self.Y[i])] = self.Y[i]

        Y_batch = torch.IntTensor(Y_batch)
        Y_lens = torch.IntTensor(Y_lens)

        return X_batch, Y_batch, Y_lens

    def _get_aug_input_lens(self):
        if not self.augment_temp:
            return self.X_lens, [[]] * self.length

        X_aug_lens = []
        X_skipped_idxs = []
        for idx in range(self.length):
            new_len = self._get_length_down_sample(self.X_lens[idx], len(self.Y[idx]))
            skipped_idxs = self._get_random_skip_idxs(new_len, len(self.Y[idx]))

            X_skipped_idxs.append(skipped_idxs)
            X_aug_lens.append(new_len - len(skipped_idxs))

        return X_aug_lens, X_skipped_idxs

    def _get_length_down_sample(self, L, out_seq_len):
        diff = self._get_aug_diff(L, out_seq_len)
        if diff < 1:
            return L

        return int(L - DOWN_SAMPLE_FACTOR * np.random.rand() * diff)

    def _get_aug_diff(self, L, out_seq_len):
        return L - out_seq_len * 4

    def _get_random_skip_idxs(self, L, out_seq_len):
        diff = self._get_aug_diff(L, out_seq_len)
        if diff < 3:
            return []

        skipped_idxs = []
        idxs = np.linspace(0, L - 1, diff + 1)

        for i in range(1, len(idxs)):
            if np.random.rand() < RANDOM_SKIP_TH:
                step = idxs[i] - idxs[i - 1]
                skip_idx = int(np.random.rand() * step + idxs[i - 1])
                if not skipped_idxs or skip_idx != skipped_idxs[-1]:
                    skipped_idxs.append(skip_idx)

        skipped_idxs.sort()
        return skipped_idxs
