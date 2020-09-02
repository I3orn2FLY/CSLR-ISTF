import sys

sys.path.append("..")
from dataset.end2end_base import *

from processing_tools import get_images, preprocess_2d, preprocess_3d
from vocab import Vocab

from utils import get_video_path


def get_video_worker(args):
    images, aug_frame, aug_temp, aug_len, skip_idxs = args

    if aug_temp:
        images = down_sample(images, aug_len + len(skip_idxs))
        images = random_skip(images, skip_idxs)

    if aug_frame:
        crop_video(images)

    video = []
    for img in images:
        if STF_TYPE == 0:
            img = preprocess_2d(img)
        else:
            img = preprocess_3d(img)
        video.append(img)

    video = np.stack(video).astype(np.float32)

    if STF_TYPE == 0:
        video = video.transpose([0, 3, 1, 2])
    else:
        video = video.transpose([3, 0, 1, 2])

    return video


class End2EndRawDataset(End2EndDataset):
    def __init__(self, vocab, split, max_batch_size, augment_frame=True, augment_temp=True):
        super(End2EndRawDataset, self).__init__(vocab, split, max_batch_size, augment_frame, augment_temp)

    def _get_ffm(self):
        return "videos"

    def _show_progress(self):
        return SHOW_PROGRESS

    def _get_feat(self, row, glosses=None):

        video_path, feat_path = get_video_path(row, self.split)

        feat = get_images(video_path)
        feat_len = len(feat)

        if feat_len < len(glosses) * 4:
            return None, None, None

        return video_path, feat, feat_len

    def get_X_batch(self, idx):
        batch_idxs = self.batches[idx]
        arg_list = []
        for i in batch_idxs:
            images = get_images(self.X[i])
            arg_list.append((images, self.augment_frame, self.augment_temp,
                             self.X_aug_lens[i], self.X_skipped_idxs[i]))

        X_batch = []
        for arg in arg_list:
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
