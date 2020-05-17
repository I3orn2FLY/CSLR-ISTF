import pickle
import shutil
from models import get_end2end_model
from common import *
from utils import *


def get_gloss_paths(images, pad_image, gloss_idx, stride, save=True):
    gloss_paths = []

    s = 0
    p = stride // 2

    images = p * [pad_image] + images + p * [pad_image]
    while s < len(images):
        e = min(len(images), s + 2 * stride)
        if e - s > stride:
            gloss_video_dir = os.path.join(GR_VIDEOS_DIR, str(gloss_idx))

            if save:
                gloss_images = images[s:e]
                if os.path.exists(gloss_video_dir):
                    shutil.rmtree(gloss_video_dir)

                os.makedirs(gloss_video_dir)

                for idx, image in enumerate(gloss_images):
                    cv2.imwrite(os.path.join(gloss_video_dir, str(idx) + ".jpg"), image)

            gloss_paths.append(os.path.join(str(gloss_idx), "*.jpg"))

            gloss_idx += 1

        s += stride

    return gloss_paths


def generate_gloss_dataset(vocab, stf_type=STF_TYPE, use_feat=USE_STF_FEAT):
    if not STF_MODEL.startswith("resnet{2+1}d") or stf_type != 1:
        print("Incorrect feature extraction model:", STF_MODEL, STF_TYPE)
        exit(0)

    print("Genearation of the Gloss-Recognition Dataset")
    model, loaded = get_end2end_model(vocab, True, True, stf_type, use_feat)

    if not loaded:
        print("STF or SEQ2SEQ model doesn't exist")
        exit(0)

    model.eval()

    pad_image = 255 * np.ones((260, 210, 3)) * np.array([0.406, 0.485, 0.456])

    pad_image = pad_image.astype(np.uint8)

    temp_stride = 4
    df = get_split_df("train")
    Y = []
    gloss_paths = []
    with torch.no_grad():

        pp = ProgressPrinter(df.shape[0], 5)
        gloss_idx = 0

        for idx in range(df.shape[0]):

            row = df.iloc[idx]

            video_path, feat_path = get_video_path(row, "train")

            images = get_images(video_path)
            if len(images) < 4:
                continue
            gloss_paths += get_gloss_paths(images, pad_image, gloss_idx, temp_stride)
            if use_feat:
                tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
            else:
                tensor_video = get_tensor_video(images, preprocess_3d, "3D").unsqueeze(0).to(DEVICE)

            preds = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1)

            for i in range(preds.size(0)):
                gloss = preds[i].item()
                Y.append(gloss)

            assert (len(Y) == len(gloss_paths))

            gloss_idx = len(Y)
            if SHOW_PROGRESS:
                pp.show(idx)

        if SHOW_PROGRESS:
            pp.end()

    Y_gloss = [vocab.idx2gloss[i] for i in Y]

    df = pd.DataFrame({"folder": gloss_paths, "gloss": Y_gloss, "gloss_idx": Y})

    L = df.shape[0]
    idxs = list(range(L))
    np.random.shuffle(idxs)
    df_train = df.iloc[idxs[:int(0.9 * L)]]
    df_val = df.iloc[idxs[int(0.9 * L):]]

    if not os.path.exists(GR_ANNO_DIR):
        os.makedirs(GR_ANNO_DIR)

    df_train.to_csv(os.path.join(GR_ANNO_DIR, "gloss_train.csv"), index=None)
    df_val.to_csv(os.path.join(GR_ANNO_DIR, "gloss_val.csv"), index=None)


class GR_dataset():
    def __init__(self, split, load, batch_size):

        self.batch_size = batch_size
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

        self.batches = [[]]

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

        X_batch = np.stack(X_batch).transpose([0, 4, 1, 2, 3])
        X_batch = torch.Tensor(X_batch)
        Y_batch = torch.LongTensor(Y_batch)

        return X_batch, Y_batch


if __name__ == "__main__":
    vocab = Vocab()
    generate_gloss_dataset(vocab, use_feat=True)

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
