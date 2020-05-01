import sys
import os
import pickle

sys.path.append(".." + os.sep)

# TODO check functions from common

from common import *
from utils import *


class GR_dataset():
    def __init__(self, split, batch_size):

        self.batch_size = batch_size
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        self.batches = [[]]

        self.build_dataset(split)

    def build_dataset(self, split):

        prefix_dir = os.sep.join([VARS_DIR, "GR_Dataset", SOURCE, END2END_TRAIN_MODE])

        X_path = os.sep.join([prefix_dir, "X_" + split + ".pkl"])
        Y_path = os.sep.join([prefix_dir, "Y_" + split + ".pkl"])
        X_lens_path = os.sep.join([prefix_dir, "X_lens_" + split + ".pkl"])

        if os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(X_lens_path):
            with open(X_path, 'rb') as f:
                self.X = pickle.load(f)

            with open(Y_path, 'rb') as f:
                self.Y = pickle.load(f)

            with open(X_lens_path, 'rb') as f:
                self.X_lens = pickle.load(f)

            print("GR", split, "dataset loaded")
            return

        print("Building GR", split, "dataset")
        df = pd.read_csv(os.path.join(ANNO_DIR, "gloss_" + split + ".csv"))
        self.X = []
        self.X_lens = []
        self.Y = []
        pp = ProgressPrinter(df.shape[0], 25)
        for i in range(df.shape[0]):
            row = df.iloc[i]
            video_dir = os.path.join(GLOSS_DATA_DIR, row.folder)
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE_3D, IMG_SIZE_3D))
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

        return len(self.batches)

    def get_batch(self, i):
        batch_idxs = self.batches[i]
        X_batch = []
        Y_batch = []
        for idx in batch_idxs:
            x, y = self.get_sample(idx)
            X_batch.append(x)
            Y_batch.append(y)

        X_batch = np.stack(X_batch).transpose([0, 4, 1, 2, 3])
        X_batch = torch.Tensor(X_batch)
        Y_batch = torch.LongTensor(Y_batch)

        return X_batch, Y_batch


if __name__ == "__main__":
    vocab = Vocab()
    gr_train = GR_dataset("train", 64)

    gr_train.start_epoch()

    X_batch, Y_batch = gr_train.get_batch(0)
    print(X_batch.size(), Y_batch.size())

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
