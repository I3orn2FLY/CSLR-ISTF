import torch
import sys
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os

sys.path.append(".." + os.sep)

from utils import *
from models import FrameFeatModel, TempFusion3D, SLR
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


def get_images(video_dir, resize=False, change_color=True):
    if SOURCE == "PH":
        image_files = list(glob.glob(video_dir))
        image_files.sort()
        images = []
        for img_file in image_files:
            img = cv2.imread(img_file)
            if change_color:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if resize:
                img = cv2.resize(img, (IMG_SIZE_3D, IMG_SIZE_3D))

            images.append(img)

    else:
        images = []
        cap = cv2.VideoCapture(video_dir)
        while True:
            ret, img = cap.read()
            if not ret:
                break

            if change_color:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if resize:
                img = cv2.resize(img, (IMG_SIZE_3D, IMG_SIZE_3D))

            images.append(img)
        cap.release()

    return images


def get_tensor_video(images, preprocess, resize=True, change_color=False):
    video = []
    for img in images:
        if resize:
            img = cv2.resize(img, (IMG_SIZE_3D, IMG_SIZE_3D))

        if change_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img.astype(np.float32))
        img = preprocess(img)
        video.append(img)

    video_tensor = torch.stack(video, dim=0)
    video_tensor = video_tensor.permute(1, 0, 2, 3)

    return video_tensor.to(DEVICE)


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
        generate_3dcnn_features_split(model, preprocess, "dev")


def save_glosses(images, gloss_idx, temporal_stride):
    gloss_paths = []
    for j in range(len(images) // temporal_stride):

        gloss_path = os.path.join(GLOSS_DATA_DIR, str(gloss_idx))
        if not os.path.exists(gloss_path):
            os.makedirs(gloss_path)

        for idx, image in enumerate(images[j * temporal_stride: (j + 1) * temporal_stride]):
            cv2.imwrite(os.path.join(gloss_path, str(idx) + ".jpg"), image)

        gloss_paths.append(os.path.join(str(gloss_idx), "*.jpg"))

        gloss_idx += 1

    return gloss_paths


def down_sample_images(images, temp_stride=4):
    L = len(images)
    n_gloss = L // temp_stride
    des_L = int(n_gloss * temp_stride)
    idxs = np.linspace(0, L - 1, des_L)
    imgs = [images[int(round(i))] for i in idxs]
    return imgs


def generate_gloss_dataset():
    vocab = Vocab()
    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=2).to(DEVICE)
    if os.path.exists(OVERFIT_END2END_MODEL_PATH):
        model.load_state_dict(torch.load(OVERFIT_END2END_MODEL_PATH, map_location=DEVICE))
        print("Model Loaded")
    else:
        print("Model doesnt exist")
        exit(0)

    model.eval()

    temp_stride = 4
    df = get_split_df("train")
    Y = []
    gloss_paths = []
    with torch.no_grad():

        pp = ProgressPrinter(df.shape[0], 5)
        gloss_idx = 0

        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

        for idx in range(df.shape[0]):

            row = df.iloc[idx]

            if SOURCE == "PH":
                video_dir = os.sep.join([VIDEOS_DIR, "train", row.folder])
            elif SOURCE == "KRSL":
                video_dir = os.path.join(VIDEOS_DIR, row.video)

            images = get_images(video_dir, resize=True, change_color=False)
            if len(images) < 4:
                continue

            images = down_sample_images(images, temp_stride=temp_stride)
            gloss_paths += save_glosses(images, gloss_idx, temp_stride)

            tensor_video = get_tensor_video(images, preprocess, resize=False, change_color=True).unsqueeze(0)
            preds = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1)

            for i in range(preds.size(0)):
                gloss = preds[i].item()
                Y.append(gloss)

            assert (len(Y) == len(gloss_paths))

            if SHOW_PROGRESS:
                pp.show(idx)

        pp.end()

    Y_gloss = [vocab.idx2gloss[i] for i in Y]

    df = pd.DataFrame({"folder": gloss_paths, "gloss": Y_gloss, "gloss_idx": Y})

    L = df.shape[0]
    idxs = list(range(L))
    np.random.shuffle(idxs)
    df_train = df.iloc[idxs[:int(0.9 * L)]]
    df_val = df.iloc[idxs[int(0.9 * L):]]

    df_train.to_csv(os.path.join(ANNO_DIR, "gloss_train.csv"), index=None)
    df_val.to_csv(os.path.join(ANNO_DIR, "gloss_val.csv"), index=None)


if __name__ == "__main__":
    # generate_cnn_features()
    generate_gloss_dataset()
    # generate_3dcnn_features()

    # generate_gloss_dataset(with_blank=False)
