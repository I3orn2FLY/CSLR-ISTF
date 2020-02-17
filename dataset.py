import pandas as pd
import pickle
import glob
import torch
import cv2
from torch.utils.data import Dataset
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


class PhoenixHandVideoDataset(Dataset):
    def __init__(self, vocab, split, augment=False):
        self.split = split
        self.augment = augment
        self.vocab = vocab
        self.df = get_pheonix_df(split)
        self.mean = np.load(os.path.join(VARS_DIR, os.path.split(HANDS_NP_IMGS_DIR)[1] + "_mean.npy"))
        self.std = np.load(os.path.join(VARS_DIR, os.path.split(HANDS_NP_IMGS_DIR)[1] + "_std.npy"))

    def _noise(self, img):
        img += 0.01 - 0.02 * random.rand(*img.shape)
        return img

    def _crop(self, img):
        img = img.transpose([1, 2, 0])
        h, w = img.shape[:2]
        y1, x1 = int(0.1 * random.rand() * h), int(0.1 * random.rand() * h)
        y2, x2 = h - int(0.1 * random.rand() * h), w - int(0.1 * random.rand() * h)
        img = img[y1:y2, x1:x2]
        img = cv2.resize(img, (w, h))
        img = img.transpose([2, 1, 0])
        return img

    def _down_sample(self, imgs, out_seq_len):
        L = len(imgs)
        diff = L // 4 - out_seq_len
        if diff < 1:
            return imgs
        n = int(L - 0.2 * random.rand() * diff)
        imgs = np.array([imgs[int(i)] for i in np.linspace(0, L - 1, n)])
        return imgs

    def _frame_skip(self, imgs, out_seq_len):
        diff = len(imgs) // 4 - out_seq_len
        if diff < 3:
            return imgs

        idxs = np.linspace(0, len(imgs) - 1, diff + 1)

        imgs = [img for img in imgs]

        skipped = 0
        for i in range(1, len(idxs)):
            if np.random.rand() < 0.5:
                step = idxs[i] - idxs[i - 1]
                skip_idx = int(np.random.rand() * step + idxs[i - 1]) - skipped
                skipped += 1
                del imgs[skip_idx]

        imgs = np.array(imgs)

        return imgs

    def _augment_imgs(self, imgs, out_seq_len):
        for i, img in enumerate(imgs):
            if random.rand() < 0.8:
                imgs[i] = self._crop(img)

            if random.rand() < 0.8:
                imgs[i] = self._noise(img)

        if random.rand() < 0.7:
            imgs = self._down_sample(imgs, out_seq_len)

        if random.rand() < 0.7:
            imgs = self._frame_skip(imgs, out_seq_len)

        return imgs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row.annotation
        vectors = self.vocab.encode(text)
        out_seq_len = len(vectors)

        video_dir = os.sep.join([HANDS_NP_IMGS_DIR, self.split, row.folder])
        np_video_file = video_dir.replace("/*.png", ".npy")
        imgs = np.load(np_video_file)
        imgs = (imgs - self.mean) / self.std

        if self.split == "train" and self.augment:
            imgs = self._augment_imgs(imgs, out_seq_len)

        imgs = torch.Tensor(imgs)
        glosses = torch.LongTensor(vectors)
        out_seq_len = torch.LongTensor([len(vectors)])

        return (imgs, glosses, out_seq_len)

    def __len__(self):
        return self.df.shape[0]


def get_pheonix_df(split):
    path = os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"])
    return pd.read_csv(path, sep='|')


def get_tensor_batch_vgg_s(X, preprocess=preprocess_vgg_s):
    batch = []
    for image_files in X:
        images = [Image.open(img_file) for img_file in image_files]

        video_tensor = torch.stack([preprocess(image) for image in images])
        batch.append(video_tensor)

    batch = torch.stack(batch)
    return torch.Tensor(batch)


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


def get_batches_vgg_s_pheonix(split, vocab, max_batch_size, shuffle, target_format=2):
    X_path = os.sep.join([VARS_DIR, "X_vgg_s_" + split + ".pkl"])

    y_path = os.sep.join([VARS_DIR, "y_vgg_s_" + split + ".pkl"])

    if os.path.exists(X_path) and os.path.exists(y_path):
        with open(X_path, 'rb') as f:
            X = pickle.load(f)

        with open(y_path, 'rb') as f:
            y = pickle.load(f)

        return X, y

    df = get_pheonix_df(split)
    X = []
    y = []
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        text = row.annotation
        img_dir = os.sep.join([HANDS_DIR, split, row.folder])

        image_files = list(glob.glob(img_dir))
        image_files.sort()

        vectors = vocab.encode(text)

        X.append(image_files)
        y.append(vectors)

    X_batches, y_batches = split_batches(X, y, max_batch_size, shuffle, target_format=target_format)

    with open(X_path, 'wb') as f:
        pickle.dump(X_batches, f)

    with open(y_path, 'wb') as f:
        pickle.dump(y_batches, f)

    return X_batches, y_batches


if __name__ == "__main__":
    vocab = Vocab()
    dataset = PhoenixHandVideoDataset(vocab, "train", augment=True)

    for i in range(5):
        imgs, glosses, seq_len = dataset[i]
    print()
