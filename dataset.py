import pickle
import torch
import cv2
import numpy as np
from feature_extraction.cnn_features import get_images
from torchvision import transforms
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

    if FRAME_FEAT_MODEL.startswith("pose"):
        dataset_class = End2EndPoseDataset
    elif FRAME_FEAT_MODEL.startswith("densenet121") or FRAME_FEAT_MODEL.startswith("googlenet"):
        dataset_class = End2EndImgFeatDataset
    elif FRAME_FEAT_MODEL.startswith("resnet{2+1}d"):
        if TEMP_FUSION_TYPE == 2:
            dataset_class = End2EndRawDataset
            args["img_size"] = 112
        else:
            dataset_class = End2EndTempFusionDataset
    elif FRAME_FEAT_MODEL.startswith("vgg-s") and END2END_TRAIN_MODE == "HAND":
        args["img_size"] = 101
        dataset_class = End2EndRawDataset

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
        ffm = FRAME_FEAT_MODEL

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

    def _get_X_batch(self, batch_idxs):

        raise NotImplementedError

    def get_batch(self, idx):
        batch_idxs = self.batches[idx]
        Y_lens = [len(self.Y[i]) for i in batch_idxs]

        X_batch = self._get_X_batch(batch_idxs)

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

    def _down_sample(self, video, n):
        video = [video[int(i)] for i in np.linspace(0, len(video) - 1, n)]
        return video

    def _random_skip(self, video, skipped_idxs):
        res_video = []

        for i in range(len(video)):
            if skipped_idxs and i == skipped_idxs[0]:
                skipped_idxs.pop(0)
                continue

            res_video.append(video[i])

        return res_video


class End2EndPoseDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not FRAME_FEAT_MODEL.startswith("pose"):
            print("Incorrect feat model:", FRAME_FEAT_MODEL)
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

    def _get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = np.load(self.X[i])
            video = process_video_pose(video, augment_frame=self.augment_frame)
            if self.augment_temp:
                video = self._down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = self._random_skip(video, self.X_skipped_idxs[i])
                video = np.stack(video)

            X_batch.append(video)

        X_batch = torch.from_numpy(np.stack(X_batch).astype(np.float32))

        return X_batch


class End2EndImgFeatDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not FRAME_FEAT_MODEL.startswith("densenet121") or not FRAME_FEAT_MODEL.startswith("googlenet"):
            print("Incorrect feat model:", FRAME_FEAT_MODEL)
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

    def _get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = torch.load(self.X[i])
            if self.augment_temp:
                video = self._down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = self._random_skip(video, self.X_skipped_idxs[i])
                video = torch.stack(video)

            X_batch.append(video)

        X_batch = torch.stack(X_batch).unsqueeze(1)

        return X_batch


class End2EndRawDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, img_size, augment_frame=True, augment_temp=True):
        # need some constraint here

        super(End2EndRawDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)
        self.img_size = img_size

        if FRAME_FEAT_MODEL.startswith("resnet{2+1}d"):
            self.tensor_preprocess = transforms.Compose([
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            ])

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

    def _get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = get_images(self.X[i])
            if self.augment_temp:
                video = self._down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = self._random_skip(video, self.X_skipped_idxs[i])

            if self.augment_frame:
                self._crop_video(video, self.img_size)
                self._noise_video(video)
            else:
                for i, img in enumerate(video):
                    video[i] = cv2.resize(img, (self.img_size, self.img_size))

                video = np.stack(video)

            video = video.astype(np.float32) / 255

            video_t = []

            for img in video:
                img_t = torch.from_numpy(img.transpose([2, 0, 1]))
                img_t = self.tensor_preprocess(img_t)
                video_t.append(img_t)

            video_t = torch.stack(video_t).permute(1, 0, 2, 3)



            X_batch.append(video_t)

        X_batch = torch.stack(X_batch)

        return X_batch

    def _noise_video(self, video):
        video = video.astype(np.float32)
        video += 2 - 4 * random.rand(*video.shape)

        video = np.maximum(video, 0)
        video = np.minimum(video, 255)

        # video = video.astype(np.uint8)
        return video

    def _crop_video(self, video, img_size):
        cropped_video = []
        for img in video:
            img = img.transpose([1, 2, 0])
            h, w = img.shape[:2]
            y1, x1 = int(0.2 * random.rand() * h), int(0.2 * random.rand() * h)
            y2, x2 = h - int(0.2 * random.rand() * h), w - int(0.2 * random.rand() * h)

            img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (img_size, img_size))

            img = img.transpose([2, 0, 1])

            cropped_video.append(img)

        return np.stack(cropped_video)


class End2EndTempFusionDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        if not FRAME_FEAT_MODEL.startswith("resnet{2+1}d") and TEMP_FUSION_TYPE != 3:
            print("Incorrect feat model:", FRAME_FEAT_MODEL, TEMP_FUSION_TYPE)
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

    def _get_X_batch(self, batch_idxs):
        X_batch = []
        for i in batch_idxs:
            video = torch.load(self.X[i])
            if self.augment_temp:
                video = self._down_sample(video, self.X_aug_lens[i] + len(self.X_skipped_idxs[i]))
                video = self._random_skip(video, self.X_skipped_idxs[i])
                video = torch.stack(video)

            X_batch.append(video)

        X_batch = torch.stack(X_batch)

        return X_batch

    def _get_aug_diff(self, L, out_seq_len):
        return L - out_seq_len


if __name__ == "__main__":
    vocab = Vocab()
    datasets = get_end2end_datasets(vocab)
    train_dataset = datasets["Train"]
    train_dataset.start_epoch()

    X_batch, Y_batch, Y_lens = train_dataset.get_batch(0)

    print(len(train_dataset.X_lens))
    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
