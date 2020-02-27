import pandas as pd
import pickle
import glob
import torch
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import random
from utils import *
from config import *

from PIL import Image
from torchvision import transforms

preprocess_vgg_s = transforms.Compose([
    transforms.Resize(101),
    transforms.CenterCrop(101),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def hand_video_collate(batch):
    videos = []
    inp_lens = []
    target_lens = []
    targets = []

    max_inp_len = 0
    max_target_len = 0
    for (video, inp_len, target, target_len) in batch:
        max_inp_len = max(max_inp_len, inp_len)
        max_target_len = max(max_target_len, target_len)
        inp_lens.append(inp_len)
        target_lens.append(target_len)
        videos.append(video)
        targets.append(target)

    inp_lens = torch.LongTensor(inp_lens)
    target_lens = torch.LongTensor(target_lens)

    inp_batch = np.zeros((len(batch), max_inp_len, *video.shape[1:]), dtype=np.float32)
    target_batch = np.zeros((len(batch), max_target_len), dtype=np.int64)

    for idx, (video, inp_len, target, target_len) in enumerate(batch):
        inp_batch[idx][:inp_len] = video
        target_batch[idx][:target_len] = target

    inp_batch = torch.from_numpy(inp_batch)

    target_batch = torch.from_numpy(target_batch)

    return inp_batch, inp_lens, target_batch, target_lens


class PhoenixHandVideoDataset():
    def __init__(self, vocab, split, augment, max_batch_size):
        self.augment = augment
        self.max_batch_size = max_batch_size
        self._build_dataset(split, vocab)

    def _build_dataset(self, split, vocab):
        self.mean = np.load(os.path.join(VARS_DIR, os.path.split(HANDS_NP_IMGS_DIR)[1] + "_mean.npy"))
        self.std = np.load(os.path.join(VARS_DIR, os.path.split(HANDS_NP_IMGS_DIR)[1] + "_std.npy"))

        X_path = os.sep.join([VARS_DIR, "PhoenixHandVideoDataset", "X_" + split + ".pkl"])
        Y_path = os.sep.join([VARS_DIR, "PhoenixHandVideoDataset", "Y_" + split + ".pkl"])
        X_lens_path = os.sep.join([VARS_DIR, "PhoenixHandVideoDataset", "X_lens_" + split + ".pkl"])

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
                np_video_file = os.sep.join([HANDS_NP_IMGS_DIR, split, row.folder]).replace("/*.png", ".npy")
                video = np.load(np_video_file)
                self.X.append(np_video_file)
                self.Y.append(glosses)
                self.X_lens.append(len(video))

            prefix_dir = os.sep.join([VARS_DIR, "PhoenixHandVideoDataset"])

            if not os.path.exists(prefix_dir):
                os.makedirs(prefix_dir)
            with open(X_path, 'wb') as f:
                pickle.dump(self.X, f)

            with open(Y_path, 'wb') as f:
                pickle.dump(self.Y, f)

            with open(X_lens_path, 'wb') as f:
                pickle.dump(self.X_lens, f)

    def get_batch(self, idx):
        batch_idxs = self.batches[idx]
        X_batch = []
        Y_lens = []
        max_target_length = 0
        for i in batch_idxs:
            video = np.load(self.X[i])
            if self.augment:
                video = self._augment_video(video, self.X_aug_lens[i])

            # for image in video:
            #     img = image.transpose([1, 2, 0])
            #     cv2.imshow("WINDOW", img)
            #     cv2.waitKey(0)

            X_batch.append(video)
            max_target_length = max(max_target_length, len(self.Y[i]))
            Y_lens.append(len(self.Y[i]))

        X_batch = torch.Tensor(np.array(X_batch))

        Y_batch = np.zeros((len(batch_idxs), max_target_length))

        for idx, i in enumerate(batch_idxs):
            Y_batch[idx][:len(self.Y[i])] = self.Y[i]


        Y_batch = torch.IntTensor(Y_batch)
        Y_lens = torch.IntTensor(Y_lens)

        return X_batch, Y_batch, Y_lens

    def start_epoch(self, shuffle=True):
        self.X_aug_lens = self._get_aug_input_lens()
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

    def _get_aug_input_lens(self):
        if not self.augment:
            return self.X_lens

        X_lens = []
        for idx in range(self.length):
            new_len = self._get_length_down_sample(self.X_lens[idx], len(self.Y[idx]))

            X_lens.append(new_len)

        return X_lens

    def _noise(self, video):
        video = video.astype(np.float32)
        video += 2 - 4 * random.rand(*video.shape)

        video = np.maximum(video, 0)
        video = np.minimum(video, 255)

        # video = video.astype(np.uint8)
        return video

    def _crop(self, img):
        img = img.transpose([1, 2, 0])
        h, w = img.shape[:2]
        y1, x1 = int(0.2 * random.rand() * h), int(0.2 * random.rand() * h)
        y2, x2 = h - int(0.2 * random.rand() * h), w - int(0.2 * random.rand() * h)

        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (w, h))

        img = img.transpose([2, 0, 1])
        return img

    def _get_length_down_sample(self, L, out_seq_len):
        diff = L - out_seq_len * 4
        if diff < 1:
            return L

        return int(L - 0.4 * random.rand() * diff)

    def _down_sample(self, video, n):
        video = np.array([video[int(i)] for i in np.linspace(0, len(video) - 1, n)])
        return video

    # def _frame_skip(self, video, out_seq_len):
    #     diff = len(video) // 4 - out_seq_len
    #     if diff < 3:
    #         return video
    #
    #     idxs = np.linspace(0, len(video) - 1, diff + 1)
    #
    #     video = [img for img in video]
    #
    #     skipped = 0
    #     for i in range(1, len(idxs)):
    #         if np.random.rand() < 0.5:
    #             step = idxs[i] - idxs[i - 1]
    #             skip_idx = int(np.random.rand() * step + idxs[i - 1]) - skipped
    #             skipped += 1
    #             del video[skip_idx]
    #
    #     video = np.array(video)
    #
    #     return video

    def _augment_video(self, video, n):
        for i, img in enumerate(video):
            video[i] = self._crop(img)

        video = self._down_sample(video, n)

        # if random.rand() < 0.7:
        #     video = self._frame_skip(video, out_seq_len)

        # video = self._noise(video)

        return video


def get_pheonix_df(split):
    path = os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"])
    return pd.read_csv(path, sep='|')


def load_gloss_dataset(with_blank=True):
    X_path = os.sep.join([VARS_DIR, "X_gloss_"])
    y_path = os.sep.join([VARS_DIR, "y_gloss_"])
    if not with_blank:
        X_path += "no_blank_"
        y_path += "no_blank_"

    X_tr = np.load(X_path + "train.npy")
    y_tr = np.load(y_path + "train.npy")
    X_dev = np.load(X_path + "dev.npy")
    y_dev = np.load(y_path + "dev.npy")

    return X_tr, y_tr, X_dev, y_dev


def read_pheonix(split, vocab, save=False, fix_shapes=False):
    suffix = "_" + FRAME_FEAT_MODEL + "_" + split
    if fix_shapes:
        suffix += "_" + str(VIDEO_SEQ_LEN)
    suffix += ".pkl"

    X_path = os.sep.join([VARS_DIR, 'X' + suffix])

    y_path = os.sep.join([VARS_DIR, 'y' + suffix])

    if os.path.exists(X_path) and os.path.exists(y_path):
        with open(X_path, 'rb') as f:
            X = pickle.load(f)

        with open(y_path, 'rb') as f:
            y = pickle.load(f)

        return X, y

    df = get_pheonix_df(split)
    X = []
    y = []
    print("Reading", split, "split")

    pp = ProgressPrinter(df.shape[0], 25)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        text = row.annotation
        feat_path = os.sep.join([VIDEO_FEAT_DIR, split, row.folder]).replace("/*.png", ".npy")
        feats = np.load(feat_path)
        if fix_shapes:
            feats = pad_features(feats, VIDEO_SEQ_LEN)

        vectors = vocab.encode(text)
        if fix_shapes:
            vectors = vectors[:MAX_OUT_LEN]
        elif split == "train" and len(vectors) > feats.shape[0] // 4:
            pp.omit()

        X.append(feats)
        y.append(vectors)

        pp.show(idx)

    pp.end()

    if save:
        with open(X_path, 'wb') as f:
            pickle.dump(X, f)

        with open(y_path, 'wb') as f:
            pickle.dump(y, f)

    return X, y


if __name__ == "__main__":
    vocab = Vocab()
    dataset = PhoenixHandVideoDataset(vocab, "train", augment=True, max_batch_size=16)

    num_batches = dataset.start_epoch()

    for i in range(num_batches):
        X_batch, Y_batch, Y_lens = dataset.get_batch(i)
        print()

    print()
