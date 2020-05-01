import multiprocessing as mp
import cv2
import glob

from end2end_base import *

sys.path.append(".." + os.sep)

from utils import *
from common import get_images


# TODO test this

def get_images_worker(video_dir):
    images = []
    if SOURCE == "PH":
        image_files = list(glob.glob(video_dir))
        image_files.sort()

        for img_file in image_files:
            img = cv2.imread(img_file)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(video_dir)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cap.release()

    return images


def get_video_worker(args):
    video_dir, img_size, mean, std, aug_frame, aug_temp, aug_len, skip_idxs = args

    images = get_images_worker(video_dir)

    if aug_temp:
        images = down_sample(images, aug_len + len(skip_idxs))
        images = random_skip(images, skip_idxs)

    if aug_frame:
        crop_video(images)

    video = []
    for img in images:
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255

        img = (img - mean) / std

        video.append(img)

    video = np.stack(video)

    video = video.transpose([3, 0, 1, 2])

    return video


class End2EndRawDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, img_size, augment_frame=True, augment_temp=True):
        # maybe implement this
        if END2END_TRAIN_MODE == "HAND" or TEMP_FUSION_TYPE == 0:
            print("Not implemented", END2END_TRAIN_MODE, IMG_FEAT_MODEL, TEMP_FUSION_TYPE)
            exit(0)
        super(End2EndRawDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)
        self.img_size = img_size

        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    def _build_dataset(self):
        print("Building", self.split, "dataset")

        ffm = "raw_videos"

        prefix_dir = os.sep.join([VARS_DIR, "End2EndDataset", SOURCE, END2END_TRAIN_MODE, ffm])

        X_path = os.sep.join([prefix_dir, "X_" + self.split + ".pkl"])
        Y_path = os.sep.join([prefix_dir, "Y_" + self.split + ".pkl"])
        X_lens_path = os.sep.join([prefix_dir, "X_lens_" + self.split + ".pkl"])

        if os.path.exists(X_path) and os.path.exists(Y_path) and os.path.exists(X_lens_path):
            with open(X_path, 'rb') as f:
                self.X = pickle.load(f)

            with open(Y_path, 'rb') as f:
                self.Y = pickle.load(f)

            with open(X_lens_path, 'rb') as f:
                self.X_lens = pickle.load(f)
        else:
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

                if SHOW_PROGRESS:
                    pp.show(idx)

                self.X.append(feat_path)
                self.Y.append(glosses)
                self.X_lens.append(feat_len)

            pp.end()
            if not os.path.exists(prefix_dir):
                os.makedirs(prefix_dir)

            with open(X_path, 'wb') as f:
                pickle.dump(self.X, f)

            with open(Y_path, 'wb') as f:
                pickle.dump(self.Y, f)

            with open(X_lens_path, 'wb') as f:
                pickle.dump(self.X_lens, f)

        self.length = len(self.X)

    def _get_feat(self, row, glosses=None):
        if SOURCE == "PH":
            video_dir = os.sep.join([VIDEOS_DIR, self.split, row.folder])
        elif SOURCE == "KRSL":
            video_dir = os.path.join(VIDEOS_DIR, row.video)
        else:
            return None, None, None

        feat = get_images(video_dir)
        feat_len = len(feat)

        if feat_len < len(glosses) * 4:
            return None, None, None

        return video_dir, feat, feat_len

    def get_X_batch(self, batch_idxs):
        if USE_MP:
            pool = mp.Pool(processes=len(batch_idxs))

            arg_list = []
            for i in batch_idxs:
                arg_list.append((self.X[i], self.img_size, self.mean, self.std,
                                 self.augment_frame, self.augment_temp,
                                 self.X_aug_lens[i], self.X_skipped_idxs[i]))

            X_batch = pool.map(get_video_worker, arg_list)
            pool.close()
            pool.join()

        else:
            X_batch = []
            for i in batch_idxs:
                arg = (self.X[i], self.img_size, self.mean, self.std,
                       self.augment_frame, self.augment_temp,
                       self.X_aug_lens[i], self.X_skipped_idxs[i])

                X_batch.append(get_video_worker(arg))

        X_batch = torch.from_numpy(np.stack(X_batch))

        return X_batch


if __name__ == "__main__":
    vocab = Vocab()
    dataset = End2EndRawDataset(vocab, "train", 4, IMG_SIZE_3D, True, True)

    dataset.start_epoch()

    X_batch, Y_batch, Y_lens = dataset.get_batch(0)

    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
