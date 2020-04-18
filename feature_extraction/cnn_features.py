import torch
import sys
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os

sys.path.append(".." + os.sep)

from utils import *
from models import FrameFeatModel
from config import *


def get_images_files(video_dir):
    if SOURCE == "PH":
        image_files = list(glob.glob(video_dir))
        image_files.sort()

        images = [Image.open(img_file) for img_file in image_files]
    else:
        images = []
        cap = cv2.VideoCapture(video_dir)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame)
            images.append(im_pil)
        cap.release()

    return images


def generate_cnn_features_split(model, device, preprocess, split, batch_size):
    with torch.no_grad():

        df = get_split_df(split)

        print(SOURCE, "Feature extraction:", split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 10)
        for idx in range(L):
            row = df.iloc[idx]
            if SOURCE == "PH":
                video_dir = os.sep.join([VIDEOS_DIR, split, row.folder])
                feat_file = os.sep.join([VIDEO_FEAT_DIR, split, row.folder]).replace("/*.png", ".npy")
            else:
                video_dir = os.path.join(VIDEOS_DIR, row.video)
                feat_file = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".npy")

            if os.path.exists(feat_file):
                pp.omit()
                continue

            feat_dir = os.path.split(feat_file)[0]

            images = get_images_files(video_dir)
            L = len(images)
            s = 0
            feats = []
            while s < L:
                e = min(L, s + batch_size)
                inp = torch.stack([preprocess(image) for image in images[s:e]])
                s = e
                inp = inp.to(device)
                feats.append(model(inp))

            feats = torch.cat(feats, dim=0).cpu().numpy()

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)

            np.save(feat_file, feats)

            if SHOW_PROGRESS:
                pp.show(idx)
            else:
                if idx % 500 == 0:
                    pp.show(idx)
                    print()

        print()


def generate_cnn_features(batch_size=512):
    if FRAME_FEAT_MODEL in ["pose"]:
        print("Incorrect feature extraction model:", FRAME_FEAT_MODEL)
        exit(0)

    device = DEVICE
    model = FrameFeatModel().to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        generate_cnn_features_split(model, device, preprocess, "train", batch_size)
        generate_cnn_features_split(model, device, preprocess, "test", batch_size)
        generate_cnn_features_split(model, device, preprocess, "dev", batch_size)


# def generate_gloss_dataset(with_blank=True):
#     vocab = Vocab()
#     device = DEVICE
#     model = SLR(rnn_hidden=512, vocab_size=vocab.size).to(device)
#     model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr.pt"])))
#     model.eval()
#
#     X_tr, y_tr = read_pheonix_cnn_feats("train", vocab, save=True)
#     X_batches, y_batches = split_batches(X_tr, y_tr, 16, shuffle=False, target_format=2)
#     stride = 4
#     X = []
#     y = []
#     with torch.no_grad():
#         pp = ProgressPrinter(len(y_batches), 10)
#         for idx in range(len(X_batches)):
#             X_batch = X_batches[idx]
#             inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
#             preds = model(inp).log_softmax(dim=2).permute(1, 0, 2).cpu().numpy().argmax(axis=2)
#             for i in range(preds.shape[0]):
#                 for j in range(len(preds[i])):
#                     feat = X_batch[i][j * stride: (j + 1) * stride]
#                     gloss = preds[i][j]
#                     if not with_blank and gloss == 0:
#                         continue
#                     X.append(feat)
#                     y.append(gloss)
#
#             pp.show(idx)
#
#     print()
#     assert len(X) == len(y), "ASD"
#
#     X = np.array(X)
#     y = np.array(y).astype(np.int32)
#     idxs = list(range(len(y)))
#     np.random.shuffle(idxs)
#     tr = int(0.9 * len(y))
#
#     X_tr = X[:tr]
#     y_tr = y[:tr]
#
#     X_dev = X[tr:]
#     y_dev = y[tr:]
#
#     X_path = os.sep.join([VARS_DIR, "X_gloss_"])
#     y_path = os.sep.join([VARS_DIR, "y_gloss_"])
#     if not with_blank:
#         X_path += "no_blank_"
#         y_path += "no_blank_"
#
#     np.save(X_path + "train", X_tr)
#     np.save(y_path + "train", y_tr)
#     np.save(X_path + "dev", X_dev)
#     np.save(y_path + "dev", y_dev)
#
#     print(X_tr.shape, y_tr.shape)
#     print(X_dev.shape, y_dev.shape)


if __name__ == "__main__":
    generate_cnn_features()

    # generate_gloss_dataset(with_blank=False)
