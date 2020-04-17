import pickle
import torch
import cv2
import numpy as np
from utils import *
from config import *


class PhoenixEnd2EndDataset():
    def __init__(self, mode, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if split == "train":
            self.augment_temp = augment_temp
            self.augment_frame = augment_frame
        else:
            self.augment_temp = False
            self.augment_frame = False

        if mode == "Full":
            self.augment_frame = False

        self.mode = mode
        self.max_batch_size = max_batch_size
        self._build_dataset(split, vocab)

    def _build_dataset(self, split, vocab):
        self.mean = np.load(os.path.join(VARS_DIR, os.path.split(PH_HANDS_NP_IMGS_DIR)[1] + "_mean.npy"))
        self.std = np.load(os.path.join(VARS_DIR, os.path.split(PH_HANDS_NP_IMGS_DIR)[1] + "_std.npy"))

        prefix_dir = os.sep.join([VARS_DIR, "Phoenix_End2EndDataset", self.mode])

        if self.mode == "Full":
            prefix_dir = os.sep.join([prefix_dir, FRAME_FEAT_MODEL])

        X_path = os.sep.join([prefix_dir, "X_" + split + ".pkl"])
        Y_path = os.sep.join([prefix_dir, "Y_" + split + ".pkl"])
        X_lens_path = os.sep.join([prefix_dir, "X_lens_" + split + ".pkl"])

        if self.mode == "Hand":
            feat_dir = os.sep.join([PH_HANDS_NP_IMGS_DIR, split])
        else:
            feat_dir = os.sep.join([VIDEO_FEAT_DIR, split])

        if os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(X_lens_path):
            with open(X_path, 'rb') as f:
                self.X = pickle.load(f)

            with open(Y_path, 'rb') as f:
                self.Y = pickle.load(f)

            with open(X_lens_path, 'rb') as f:
                self.X_lens = pickle.load(f)

            self.length = len(self.X)

        else:
            df = get_pheonix_df(split)
            self.length = df.shape[0]
            self.X = []
            self.Y = []
            self.X_lens = []
            for idx in range(self.length):
                row = df.iloc[idx]
                glosses = vocab.encode(row.annotation)
                np_video_file = os.sep.join([feat_dir, row.folder]).replace("/*.png", ".npy")
                video = np.load(np_video_file)
                self.X.append(np_video_file)
                self.Y.append(glosses)
                self.X_lens.append(len(video))

            if not os.path.exists(prefix_dir):
                os.makedirs(prefix_dir)

            with open(X_path, 'wb') as f:
                pickle.dump(self.X, f)

            with open(Y_path, 'wb') as f:
                pickle.dump(self.Y, f)

            with open(X_lens_path, 'wb') as f:
                pickle.dump(self.X_lens, f)

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
            random.shuffle(lenghts)

        for l in lenghts:
            idxs = len_table[l]
            if shuffle:
                random.shuffle(idxs)
            s = 0
            while (s < len(idxs)):
                e = min(s + self.max_batch_size, len(idxs))

                self.batches.append(idxs[s:e])

                s += self.max_batch_size

        return len(self.batches)

    def get_batch(self, idx):
        batch_idxs = self.batches[idx]
        X_batch = []
        Y_lens = []
        for i in batch_idxs:
            video = np.load(self.X[i])

            # if FRAME_FEAT_MODEL == "pose":
            #     video = video.reshape(-1, 137, 3)

            video = self._augment_video(video, self.X_aug_lens[i], self.X_skipped_idxs[i])

            # for image in video:
            #     img = image.transpose([1, 2, 0])
            #     cv2.imshow("WINDOW", img)
            #     cv2.waitKey(0)

            X_batch.append(video)
            Y_lens.append(len(self.Y[i]))

        X_batch = torch.Tensor(np.stack(X_batch))
        if self.mode == "Full":
            X_batch = X_batch.unsqueeze(1)

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

    def _noise_video(self, video):
        video = video.astype(np.float32)
        video += 2 - 4 * random.rand(*video.shape)

        video = np.maximum(video, 0)
        video = np.minimum(video, 255)

        # video = video.astype(np.uint8)
        return video

    def _crop_video(self, video):
        for i, img in enumerate(video):
            img = img.transpose([1, 2, 0])
            h, w = img.shape[:2]
            y1, x1 = int(0.2 * random.rand() * h), int(0.2 * random.rand() * h)
            y2, x2 = h - int(0.2 * random.rand() * h), w - int(0.2 * random.rand() * h)

            img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (w, h))

            img = img.transpose([2, 0, 1])

            video[i] = img

        return video

    def _get_length_down_sample(self, L, out_seq_len):
        diff = L - out_seq_len * 4
        if diff < 1:
            return L

        return int(L - DOWN_SAMPLE_FACTOR * random.rand() * diff)

    def _get_random_skip_idxs(self, L, out_seq_len):
        diff = L - out_seq_len * 4
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

    def _down_sample(self, video, n):
        video = np.array([video[int(i)] for i in np.linspace(0, len(video) - 1, n)])
        return video

    def _random_skip(self, video, skipped_idxs):
        res_video = []

        for i in range(len(video)):
            if skipped_idxs and i == skipped_idxs[0]:
                skipped_idxs.pop(0)
                continue

            res_video.append(video[i])

        res_video = np.stack(res_video)

        return res_video

    def _augment_video(self, video, n, skipped_idxs):

        if self.augment_temp:
            video = self._down_sample(video, n + len(skipped_idxs))
            video = self._random_skip(video, skipped_idxs)

        if self.augment_frame:
            video = self._crop_video(video)
            video = self._noise_video(video)

        return video


class KRSLEnd2EndDataset():
    def __init__(self, mode, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if split == "train":
            self.augment_temp = augment_temp
            self.augment_frame = augment_frame
        else:
            self.augment_temp = False
            self.augment_frame = False

        if mode == "Full":
            self.augment_frame = False

        self.mode = mode
        self.max_batch_size = max_batch_size
        self._build_dataset(split, vocab)

    def _build_dataset(self, split, vocab):
        # self.mean = np.load(os.path.join(VARS_DIR, os.path.split(PH_HANDS_NP_IMGS_DIR)[1] + "_mean.npy"))
        # self.std = np.load(os.path.join(VARS_DIR, os.path.split(PH_HANDS_NP_IMGS_DIR)[1] + "_std.npy"))

        prefix_dir = os.sep.join([VARS_DIR, "KRSL_End2EndDataset", self.mode])

        if self.mode == "Full":
            prefix_dir = os.sep.join([prefix_dir, FRAME_FEAT_MODEL])

        X_path = os.sep.join([prefix_dir, "X_" + split + ".pkl"])
        Y_path = os.sep.join([prefix_dir, "Y_" + split + ".pkl"])
        X_lens_path = os.sep.join([prefix_dir, "X_lens_" + split + ".pkl"])

        if self.mode == "Hand":
            feat_dir = KRSL_HANDS_NP_IMGS_DIR
        else:
            feat_dir = VIDEO_FEAT_DIR

        if os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(X_lens_path):
            with open(X_path, 'rb') as f:
                self.X = pickle.load(f)

            with open(Y_path, 'rb') as f:
                self.Y = pickle.load(f)

            with open(X_lens_path, 'rb') as f:
                self.X_lens = pickle.load(f)
        else:
            df = get_KRSL_df(split)

            self.X = []
            self.Y = []
            self.X_lens = []
            for idx in range(df.shape[0]):
                row = df.iloc[idx]
                glosses = vocab.encode(row.annotation)
                np_feat_file = os.path.join(feat_dir, row.video).replace(".mp4", ".npy")
                feat = np.load(np_feat_file)
                if len(feat) < len(glosses) * 4:
                    continue
                self.X.append(np_feat_file)
                self.Y.append(glosses)
                self.X_lens.append(len(feat))

            if not os.path.exists(prefix_dir):
                os.makedirs(prefix_dir)

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
            random.shuffle(lenghts)

        for l in lenghts:
            idxs = len_table[l]
            if shuffle:
                random.shuffle(idxs)
            s = 0
            while (s < len(idxs)):
                e = min(s + self.max_batch_size, len(idxs))

                self.batches.append(idxs[s:e])

                s += self.max_batch_size

        return len(self.batches)

    def get_batch(self, idx):
        batch_idxs = self.batches[idx]
        X_batch = []
        Y_lens = []
        for i in batch_idxs:
            video = np.load(self.X[i])

            # if FRAME_FEAT_MODEL == "pose":
            #     video = video.reshape(-1, 137, 3)

            video = self._augment_video(video, self.X_aug_lens[i], self.X_skipped_idxs[i])

            # for image in video:
            #     img = image.transpose([1, 2, 0])
            #     cv2.imshow("WINDOW", img)
            #     cv2.waitKey(0)

            X_batch.append(video)
            Y_lens.append(len(self.Y[i]))

        X_batch = torch.Tensor(np.stack(X_batch))
        if self.mode == "Full":
            X_batch = X_batch.unsqueeze(1)

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

    def _noise_video(self, video):
        video = video.astype(np.float32)
        video += 2 - 4 * random.rand(*video.shape)

        video = np.maximum(video, 0)
        video = np.minimum(video, 255)

        # video = video.astype(np.uint8)
        return video

    def _crop_video(self, video):
        for i, img in enumerate(video):
            img = img.transpose([1, 2, 0])
            h, w = img.shape[:2]
            y1, x1 = int(0.2 * random.rand() * h), int(0.2 * random.rand() * h)
            y2, x2 = h - int(0.2 * random.rand() * h), w - int(0.2 * random.rand() * h)

            img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (w, h))

            img = img.transpose([2, 0, 1])

            video[i] = img

        return video

    def _get_length_down_sample(self, L, out_seq_len):
        diff = L - out_seq_len * 4
        if diff < 1:
            return L

        return int(L - DOWN_SAMPLE_FACTOR * random.rand() * diff)

    def _get_random_skip_idxs(self, L, out_seq_len):
        diff = L - out_seq_len * 4
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

    def _down_sample(self, video, n):
        video = np.array([video[int(i)] for i in np.linspace(0, len(video) - 1, n)])
        return video

    def _random_skip(self, video, skipped_idxs):
        res_video = []

        for i in range(len(video)):
            if skipped_idxs and i == skipped_idxs[0]:
                skipped_idxs.pop(0)
                continue

            res_video.append(video[i])

        res_video = np.stack(res_video)

        return res_video

    def _augment_video(self, video, n, skipped_idxs):

        if self.augment_temp:
            video = self._down_sample(video, n + len(skipped_idxs))
            video = self._random_skip(video, skipped_idxs)

        if self.augment_frame:
            video = self._crop_video(video)
            video = self._noise_video(video)

        return video


if __name__ == "__main__":
    vocab = Vocab()
    krsl_train = KRSLEnd2EndDataset(END2END_TRAIN_MODE, vocab, "train", 32)
    krsl_train.start_epoch()

    X_batch, Y_batch, Y_lens = krsl_train.get_batch(0)

    print(pd.Series(krsl_train.X_lens).value_counts())
    c = 0
    for x, y in zip(krsl_train.X_lens, krsl_train.Y):
        if x < 4 * len(y):
            c += 1

    print(c)

    print(len(krsl_train.X_lens))
    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
