from config import *
import numpy as np
import pickle
import glob
import cv2
import torch
import pandas as pd

from utils import ProgressPrinter
from vocab import Vocab


# Change this for 2D + 1D stf type

class GR_dataset():
    def __init__(self, split, load, batch_size, stf_type=STF_TYPE):

        self.batch_size = batch_size
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        self.batches = [[]]

        self.stf_type = stf_type
        self.build_dataset(split, load)

    def build_dataset(self, split, load):

        prefix_dir = os.path.join(GR_DATASET_DIR, "VARS")

        X_path = os.sep.join([prefix_dir, "X_" + split + ".pkl"])
        Y_path = os.sep.join([prefix_dir, "Y_" + split + ".pkl"])
        X_lens_path = os.sep.join([prefix_dir, "X_lens_" + split + ".pkl"])

        if load and os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(X_lens_path):
            with open(X_path, 'rb') as f:
                self.X = pickle.load(f)

            with open(Y_path, 'rb') as f:
                self.Y = pickle.load(f)

            with open(X_lens_path, 'rb') as f:
                self.X_lens = pickle.load(f)

            print("GR", split, "dataset loaded")
            return

        print("Building GR", split, "dataset")
        df = pd.read_csv(os.path.join(GR_ANNO_DIR, "gloss_" + split + ".csv"))
        self.X = []
        self.X_lens = []
        self.Y = []
        pp = ProgressPrinter(df.shape[0], 25)
        for i in range(df.shape[0]):
            row = df.iloc[i]
            video_dir = os.path.join(GR_VIDEOS_DIR, row.folder)
            image_files = list(glob.glob(video_dir))
            image_files.sort()

            self.X.append(image_files)
            self.X_lens.append(len(image_files))
            self.Y.append(int(row.gloss_idx))

            if SHOW_PROGRESS:
                pp.show(i)

        if SHOW_PROGRESS:
            pp.end()

        if not os.path.exists(prefix_dir):
            os.makedirs(prefix_dir)

        with open(X_path, 'wb') as f:
            pickle.dump(self.X, f)

        with open(Y_path, 'wb') as f:
            pickle.dump(self.Y, f)

        with open(X_lens_path, 'wb') as f:
            pickle.dump(self.X_lens, f)

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE_2Plus1D, IMG_SIZE_2Plus1D))
            img = img.astype(np.float32) / 255
            img = (img - self.mean) / self.std
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

    ls = list(glob.glob(GR_VIDEOS_DIR + "/*"))

    # print(len(ls))
    # gr_train = GR_dataset("train", False, 64)
    #
    # n = gr_train.start_epoch()
    #
    # pp = ProgressPrinter(n, 5)
    #
    # lengths = {}
    # for i in range(n):
    #     X_batch, Y_batch = gr_train.get_batch(i)
    #     L = X_batch.size(2)
    #     lengths[L] = lengths.get(L, 0) + 1
    #     pp.show(i)
    # pp.end()
    # print(lengths)

    # df = pd.read_csv(os.path.join(GR_ANNO_DIR, "gloss_" + split + ".csv"))
    #
    # pp = ProgressPrinter(df.shape[0], 25)
    # for i in range(df.shape[0]):
    #     row = df.iloc[i]
    #     video_dir = os.path.join(GR_VIDEOS_DIR, row.folder)
    #     image_files = list(glob.glob(video_dir))
    #     image_files.sort()
    #
    #     if SHOW_PROGRESS:
    #         pp.show(i)

# gr_train.start_epoch()
# idxs = gr_train.batches[0]
# X_batch, Y_batch = gr_train.get_batch(0)
# X_batch = X_batch.numpy()
# print(idxs)
# X_batch = X_batch.transpose([0, 2, 3, 4, 1])
# mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
# std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
#
# for vid in X_batch:
#     vid = (vid * std + mean) * 255
#     vid = vid.astype(np.uint8)
#     for image in vid:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         cv2.imshow("window", image)
#         if cv2.waitKey(0) == 27:
#             exit(0)
#
# print(X_batch.shape)
