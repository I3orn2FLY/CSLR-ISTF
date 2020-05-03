import multiprocessing as mp

from end2end_base import *

sys.path.append(".." + os.sep)

from common import get_images, preprocess_2d, preprocess_3d

from utils import Vocab


def get_video_worker(args):
    video_dir, mean, std, aug_frame, aug_temp, aug_len, skip_idxs = args

    images = get_images(video_dir)

    if aug_temp:
        images = down_sample(images, aug_len + len(skip_idxs))
        images = random_skip(images, skip_idxs)

    if aug_frame:
        crop_video(images)

    video = []
    for img in images:
        if TEMP_FUSION_TYPE == 0:
            img = preprocess_2d(img)
        else:
            img = preprocess_3d(img)
        video.append(img)

    video = np.stack(video).astype(np.float32)

    if TEMP_FUSION_TYPE == 0:
        video = video.transpose([0, 3, 1, 2])
    else:
        video = video.transpose([3, 0, 1, 2])

    return video


class End2EndRawDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        # maybe implement this
        if USE_FEAT:
            print("Error, using Features")
            exit(0)
        super(End2EndRawDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    def _get_ffm(self):
        return "videos"

    def _show_progress(self):
        return SHOW_PROGRESS

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
                arg_list.append((self.X[i], self.mean, self.std,
                                 self.augment_frame, self.augment_temp,
                                 self.X_aug_lens[i], self.X_skipped_idxs[i]))

            X_batch = pool.map(get_video_worker, arg_list)
            pool.close()
            pool.join()

        else:
            X_batch = []
            for i in batch_idxs:
                arg = (self.X[i], self.mean, self.std,
                       self.augment_frame, self.augment_temp,
                       self.X_aug_lens[i], self.X_skipped_idxs[i])

                X_batch.append(get_video_worker(arg))

        X_batch = torch.from_numpy(np.stack(X_batch))

        return X_batch


if __name__ == "__main__":
    vocab = Vocab()
    dataset = End2EndRawDataset(vocab, "train", 4, True, True)

    dataset.start_epoch()

    X_batch, Y_batch, Y_lens = dataset.get_batch(0)

    print(X_batch.size())
    print(Y_batch.size())
    print(Y_lens.size())
