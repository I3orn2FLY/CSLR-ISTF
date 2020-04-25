import torch
import sys
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os

sys.path.append(".." + os.sep)

from utils import *
from models import FrameFeatModel, TempFusion3D
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


def get_images(video_dir):
    if SOURCE == "PH":
        image_files = list(glob.glob(video_dir))
        image_files.sort()
        images = [cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB) for img_file in image_files]
    else:
        images = []
        cap = cv2.VideoCapture(video_dir)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(img)
        cap.release()

    return images


def get_tensor_video(images, preprocess):
    video = []
    for img in images:
        img = cv2.resize(img, (IMG_SIZE_3D, IMG_SIZE_3D)) / 255
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img.astype(np.float32))
        img = preprocess(img)
        video.append(img)

    video_tensor = torch.stack(video, dim=0).to(DEVICE)
    video_tensor = video_tensor.permute(1, 0, 2, 3)

    return video_tensor


def generate_cnn_features_split(model, device, preprocess, split, batch_size):
    with torch.no_grad():
        if SOURCE == "KRSL" and split == "dev":
            split = "val"

        df = get_split_df(split)

        print(SOURCE, FRAME_FEAT_MODEL, "feature extraction:", split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 10)
        for idx in range(L):
            row = df.iloc[idx]
            if SOURCE == "PH":
                video_dir = os.sep.join([VIDEOS_DIR, split, row.folder])
                feat_path = os.sep.join([VIDEO_FEAT_DIR, split, row.folder]).replace("/*.png", ".pt")
            else:
                video_dir = os.path.join(VIDEOS_DIR, row.video)
                feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")

            if os.path.exists(feat_path):
                pp.omit()
                continue

            feat_dir = os.path.split(feat_path)[0]

            images = get_images_files(video_dir)
            if not images:
                continue
            L = len(images)
            s = 0
            video_feat = []
            while s < L:
                e = min(L, s + batch_size)
                inp = torch.stack([preprocess(image) for image in images[s:e]])
                s = e
                inp = inp.to(device)
                video_feat.append(model(inp))

            video_feat = torch.cat(video_feat, dim=0).cpu()

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)


            torch.save(video_feat, feat_path)

            if SHOW_PROGRESS:
                pp.show(idx)
            else:
                if idx % 500 == 0:
                    pp.show(idx)
                    print()

        print()


def generate_cnn_features(batch_size=FEAT_EX_BATCH_SIZE):
    if FRAME_FEAT_MODEL.startswith("pose") or FRAME_FEAT_MODEL.startswith("resnet{2+1}d"):
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


def generate_3dcnn_features_split(model, preprocess, split):
    with torch.no_grad():
        if SOURCE == "KRSL" and split == "dev":
            split = "val"

        df = get_split_df(split)

        print(SOURCE, FRAME_FEAT_MODEL, "feature extraction:", split, "split")
        L = df.shape[0]

        pp = ProgressPrinter(L, 10)
        for idx in range(L):
            row = df.iloc[idx]
            if SOURCE == "PH":
                video_dir = os.sep.join([VIDEOS_DIR, split, row.folder])
                feat_path = os.sep.join([VIDEO_FEAT_DIR, split, row.folder]).replace("/*.png", ".pt")
            else:
                video_dir = os.path.join(VIDEOS_DIR, row.video)
                feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")

            if os.path.exists(feat_path):
                pp.omit()
                continue

            feat_dir = os.path.split(feat_path)[0]

            images = get_images(video_dir)
            if len(images) < 4:
                continue

            tensor_video = get_tensor_video(images, preprocess)
            feat = model(tensor_video.unsqueeze(0)).squeeze(0).cpu()

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)

            torch.save(feat, feat_path)

            if SHOW_PROGRESS:
                pp.show(idx)
            else:
                if idx % 500 == 0:
                    pp.show(idx)
                    print()

        print()


def generate_3dcnn_features():
    if not FRAME_FEAT_MODEL.startswith("resnet{2+1}d"):
        print("Incorrect feature extraction model:", FRAME_FEAT_MODEL)
        exit(0)

    model = TempFusion3D().to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])
    with torch.no_grad():
        generate_3dcnn_features_split(model, preprocess, "train")
        generate_3dcnn_features_split(model, preprocess, "test")
        generate_3dcnn_features_split(model, preprocess, "dev", )


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
    # generate_3dcnn_features()

    # generate_gloss_dataset(with_blank=False)
