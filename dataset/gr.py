from config import *
import numpy as np
import pickle
import glob
import cv2
import torch
import pandas as pd

from utils import ProgressPrinter
from processing_tools import preprocess_3d, preprocess_2d
from vocab import Vocab


# TODO update GR dataset without csv files

class GR_dataset():
    def __init__(self, split, batch_size, stf_type=STF_TYPE):

        self.batch_size = batch_size
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        self.batches = [[]]

        self.stf_type = stf_type
        self.load_dataset(split)

    def load_dataset(self, split):
        data_path = os.sep.join([GR_DATASET_DIR, "VARS", "data.pkl"])
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            X = data["X"]
            Y = data["Y"]
            X_lens = data["X_lens"]
            idxs = data["idxs"]

            if split == "train":
                idxs = idxs[:int(0.9 * len(X))]
            else:
                idxs = idxs[int(0.9 * len(X)):]

            self.X = [X[idx] for idx in idxs]
            self.Y = [Y[idx] for idx in idxs]
            self.X_lens = [X_lens[idx] for idx in idxs]

            print("GR", split, "dataset loaded")
        else:
            raise ValueError("GR Dataset not generated!")

    def get_sample(self, i):
        y = self.Y[i]
        image_files = self.X[i]
        images = []
        for img_file in image_files:
            img = cv2.imread(img_file)
            h, w = img.shape[:2]
            y1, x1 = int(0.2 * np.random.rand() * h), int(0.2 * np.random.rand() * h)
            y2, x2 = h - int(0.2 * np.random.rand() * h), w - int(0.2 * np.random.rand() * h)
            img = img[y1:y2, x1:x2]
            if self.stf_type == 1:
                img = preprocess_3d(img)
            else:
                img = preprocess_2d(img)
            images.append(img)

        x = np.stack(images)
        return x, y

    def start_epoch(self, shuffle=True):
        len_table = dict()

        for i, length in enumerate(self.X_lens):
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
                e = min(s + self.batch_size, len(idxs))

                self.batches.append(idxs[s:e])

                s += self.batch_size

        if shuffle:
            np.random.shuffle(self.batches)
        return len(self.batches)

    def get_batch(self, i):
        batch_idxs = self.batches[i]
        X_batch = []
        Y_batch = []
        for idx in batch_idxs:
            x, y = self.get_sample(idx)
            X_batch.append(x)
            Y_batch.append(y)

        X_batch = np.stack(X_batch)
        if self.stf_type == 0:
            X_batch = X_batch.transpose([0, 1, 4, 2, 3])
        else:
            X_batch = X_batch.transpose([0, 4, 1, 2, 3])

        X_batch = torch.Tensor(X_batch)
        Y_batch = torch.LongTensor(Y_batch)

        return X_batch, Y_batch


if __name__ == "__main__":
    vocab = Vocab()
    gr_train = GR_dataset("train", 64, 0)

    n = gr_train.start_epoch()
    X_batch, Y_batch = gr_train.get_batch(0)

    print(X_batch.shape, Y_batch.shape)
