import pickle
import torch
import multiprocessing as mp
import cv2
import numpy as np
from feature_extraction.cnn_features import get_images
from utils import *
from config import *


# TODO END2END Hand testing and fixing
# TODO test End2EndImgfeat
# TODO test End2EndImgRaw
# TODO test End2EndImgRaw

def process_video_pose(video_pose, augment_frame=True):
    video_pose = video_pose.reshape(-1, 137, 3)
    idxs = []
    video_pose = video_pose[:, :, :2]

    noise = []
    if POSE_FACE:
        idxs += list(range(70))
        noise.append(POSE_AUG_NOISE_HANDFACE - 2 * POSE_AUG_NOISE_HANDFACE * random.rand(len(video_pose), 70, 2))

    if POSE_BODY:
        idxs += list(range(70, 70 + 8)) + list(range(70 + 15, 70 + 19))
        noise.append(POSE_AUG_NOISE_BODY - 2 * POSE_AUG_NOISE_BODY * random.rand(len(video_pose), 12, 2))

    if POSE_HANDS:
        idxs += list(range(95, 137))

        noise.append(POSE_AUG_NOISE_HANDFACE - 2 * POSE_AUG_NOISE_HANDFACE * random.rand(len(video_pose), 42, 2))

    video_pose = video_pose[:, idxs]

    if augment_frame:
        noise = np.concatenate(noise, axis=1)

        offset = POSE_AUG_OFFSET - 2 * POSE_AUG_OFFSET * random.rand(2)

    if augment_frame:
        video_pose += noise + offset

    return video_pose.reshape(len(video_pose), -1)


def get_end2end_datasets(vocab, include_test=False):
    args = {"vocab": vocab, "split": "train", "max_batch_size": END2END_BATCH_SIZE,
            "augment_temp": END2END_DATA_AUG_TEMP, "augment_frame": END2END_DATA_AUG_FRAME}

    if INP_FEAT:
        if IMG_FEAT_MODEL.startswith("pose"):
            dataset_class = End2EndPoseDataset
        elif IMG_FEAT_MODEL.startswith("densenet121") or IMG_FEAT_MODEL.startswith("googlenet"):
            dataset_class = End2EndImgFeatDataset
        elif IMG_FEAT_MODEL.startswith("resnet{2+1}d"):
            dataset_class = End2EndTempFusionDataset
        else:
            print("Not implemented", IMG_FEAT_MODEL, TEMP_FUSION_TYPE)
            exit(0)
    else:
        dataset_class = End2EndRawDataset
        if TEMP_FUSION_TYPE == 0:
            args["img_size"] = IMG_SIZE_2D
        elif TEMP_FUSION_TYPE == 1:
            args["img_size"] = IMG_SIZE_3D
        else:
            print("Incorrect temp fusion type", TEMP_FUSION_TYPE)
            exit(0)

    tr_dataset = dataset_class(**args)
    args["split"] = "dev"
    val_dataset = dataset_class(**args)

    datasets = {"Train": tr_dataset, "Val": val_dataset}
    if include_test:
        args["split"] = "test"
        datasets["Test"] = dataset_class(**args)

    return datasets


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

    def _build_dataset(self):
        print("Building", self.split, "dataset")

        # self.mean = np.load(os.path.join(VARS_DIR, os.path.split(PH_HANDS_NP_IMGS_DIR)[1] + "_mean.npy"))
        # self.std = np.load(os.path.join(VARS_DIR, os.path.split(PH_HANDS_NP_IMGS_DIR)[1] + "_std.npy"))
        ffm = IMG_FEAT_MODEL

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
            for idx in range(df.shape[0]):
                row = df.iloc[idx]
                glosses = self.vocab.encode(row.annotation)
                feat_path, feat, feat_len = self._get_feat(row, glosses)
                if feat is None:
                    continue

                self.X.append(feat_path)
                self.Y.append(glosses)
                self.X_lens.append(feat_len)

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

        return int(L - DOWN_SAMPLE_FACTOR * random.rand() * diff)

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


class End2EndPoseDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not IMG_FEAT_MODEL.startswith("pose"):
            print("Incorrect feat model:", IMG_FEAT_MODEL)
            exit(0)
        super(End2EndPoseDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_feat(self, row, glosses=None):
        if SOURCE == "PH":
            feat_path = os.sep.join([VIDEO_FEAT_DIR, self.split, row.folder]).replace("/*.png", ".npy")
        elif SOURCE == "KRSL":
            feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".npy")
        else:
            return None, None, None

        if not os.path.exists(feat_path):
            return None, None, None

        feat = np.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) * 4:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = np.load(self.X[i])
            video = process_video_pose(video, augment_frame=self.augment_frame)
            if self.augment_temp:
                video = down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = random_skip(video, self.X_skipped_idxs[i])
                video = np.stack(video)

            X_batch.append(video)

        X_batch = torch.from_numpy(np.stack(X_batch).astype(np.float32)).unsqueeze(1)

        return X_batch


class End2EndImgFeatDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not IMG_FEAT_MODEL.startswith("densenet121") or not IMG_FEAT_MODEL.startswith("googlenet"):
            print("Incorrect feat model:", IMG_FEAT_MODEL)
            exit(0)
        super(End2EndImgFeatDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_feat(self, row, glosses=None):
        if SOURCE == "PH":
            feat_path = os.sep.join([VIDEO_FEAT_DIR, self.split, row.folder]).replace("/*.png", ".pt")
        elif SOURCE == "KRSL":
            feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")
        else:
            return None, None, None

        if not os.path.exists(feat_path):
            return None, None, None

        feat = torch.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) * 4:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = torch.load(self.X[i])
            if self.augment_temp:
                video = down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = random_skip(video, self.X_skipped_idxs[i])
                video = torch.stack(video)

            X_batch.append(video)

        X_batch = torch.stack(X_batch).unsqueeze(1)

        return X_batch


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


def crop_video(video):
    cropped_video = []
    for img in video:
        h, w = img.shape[:2]
        y1, x1 = int(0.2 * random.rand() * h), int(0.2 * random.rand() * h)
        y2, x2 = h - int(0.2 * random.rand() * h), w - int(0.2 * random.rand() * h)
        img = img[y1:y2, x1:x2]
        cropped_video.append(img)

    return cropped_video


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

        feat = get_images(video_dir, change_color=False)
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


# def noise_video(video):
#     video = video.astype(np.float32)
#     video += 2 - 4 * random.rand(*video.shape)
#
#     video = np.maximum(video, 0)
#     video = np.minimum(video, 255)
#
#     # video = video.astype(np.uint8)
#     return video


class End2EndTempFusionDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not IMG_FEAT_MODEL.startswith("resnet{2+1}d") and TEMP_FUSION_TYPE != 3:
            print("Incorrect feat model:", IMG_FEAT_MODEL, TEMP_FUSION_TYPE)
            exit(0)
        super(End2EndTempFusionDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_feat(self, row, glosses=None):
        if SOURCE == "PH":
            feat_path = os.sep.join([VIDEO_FEAT_DIR, self.split, row.folder]).replace("/*.png", ".pt")
        elif SOURCE == "KRSL":
            feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")
        else:
            return None, None, None

        if not os.path.exists(feat_path):
            return None, None, None

        feat = torch.load(feat_path)
        feat_len = len(feat)

        if feat_len < len(glosses) or len(feat.shape) < 2:
            return None, None, None

        return feat_path, feat, feat_len

    def get_X_batch(self, batch_idxs):
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


class GR_dataset():
    def __init__(self, split, batch_size):
        self.df = pd.read_csv(os.path.join(ANNO_DIR, "gloss_" + split + ".csv"))

        self.batch_size = batch_size
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        self.batches = [[]]

    def get_sample(self, i):
        row = self.df.iloc[i]
        y = int(row.gloss_idx)
        video_dir = os.path.join(GLOSS_DATA_DIR, row.folder)
        images = []

        image_files = list(glob.glob(video_dir))
        image_files.sort()

        for img_file in image_files:
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE_3D, IMG_SIZE_3D))
            img = img.astype(np.float32) / 255
            img = (img - self.mean) / self.std
            images.append(img)

        x = np.stack(images)

        return x, y

    def start_epoch(self):
        idxs = list(range(self.df.shape[0]))
        np.random.shuffle(idxs)
        self.batches = []
        s = 0
        while s < len(idxs):
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


def get_gr_datasets(batch_size=GR_BATCH_SIZE):
    datasets = dict()
    datasets["Train"] = GR_dataset("train", batch_size)
    datasets["Val"] = GR_dataset("val", batch_size)

    return datasets


if __name__ == "__main__":
    vocab = Vocab()
    # datasets = get_end2end_datasets(vocab)
    # train_dataset = datasets["Train"]
    # train_dataset.start_epoch()
    #
    # X_batch, _, _ = train_dataset.get_batch(0)

    gr_train = GR_dataset("train", 64)

    gr_train.start_epoch()

    X_batch, Y_batch = gr_train.get_batch(0)
    print(X_batch.size(), Y_batch.size())
