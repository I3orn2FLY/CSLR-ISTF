import torch
from torchvision import transforms
import glob
from PIL import Image
import time
import numpy as np
import pandas as pd
from models import FrameFeatModel, SLR
from config import *
from utils import print_progress
from data import split_batches, Vocab
from data import read_pheonix
import ctcdecode


def generate_split(model, device, preprocess, split):
    df = pd.read_csv(os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"]), sep='|')

    print("Feature extraction:", split, "split")
    with torch.no_grad():
        L = df.shape[0]
        start_time = time.time()
        rand_idx = int(np.random.rand() * L)
        for idx in range(L):
            row = df.iloc[idx]
            img_dir = os.sep.join([IMAGES_DIR, split, row.folder])
            feat_dir = os.sep.join([VIDEO_FEAT_DIR, split, row.folder])
            feat_file = feat_dir.replace("/*.png", "")

            if os.path.exists(feat_file + ".npy"):
                continue

            feat_dir = os.path.split(feat_file)[0]

            image_files = list(glob.glob(img_dir))
            image_files.sort()

            # image_files = [os.path.split(img_file)[1] for img_file in image_files]
            # if idx == rand_idx:
            #     print()
            #     for img_file in image_files:
            #         print(img_file.split('.')[1].split('_')[-1])
            #
            #     exit()

            images = [Image.open(img_file) for img_file in image_files]
            inp = torch.stack([preprocess(image) for image in images])
            inp = inp.to(device)
            feats = model(inp).cpu().numpy()

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)
            np.save(feat_file, feats)

            if idx % 10 == 0:
                print_progress(idx + 1, L, start_time)

        print()


def extract_features():
    device = torch.device("cuda:0")
    model = FrameFeatModel().to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    generate_split(model, device, preprocess, "train")
    generate_split(model, device, preprocess, "test")
    generate_split(model, device, preprocess, "dev")


def generate_gloss_dataset(with_blank=True):
    vocab = Vocab(source="pheonix")
    device = torch.device("cuda:0")
    model = SLR(rnn_hidden=512, vocab_size=vocab.size).to(device)
    model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr.pt"])))
    model.eval()

    X_tr, y_tr = read_pheonix("train", vocab, save=True)
    X_batches, y_batches = split_batches(X_tr, y_tr, 16, shuffle=False, target_format=2)
    stride = 4
    X = []
    y = []
    with torch.no_grad():
        start_time = time.time()
        for idx in range(len(X_batches)):
            X_batch = X_batches[idx]
            inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
            preds = model(inp).log_softmax(dim=2).permute(1, 0, 2).cpu().numpy().argmax(axis=2)
            for i in range(preds.shape[0]):
                for j in range(len(preds[i])):
                    feat = X_batch[i][j * stride: (j + 1) * stride]
                    gloss = preds[i][j]
                    if not with_blank and gloss == 0:
                        continue
                    X.append(feat)
                    y.append(gloss)

            if idx % 5 == 0:
                print_progress(idx + 1, len(X_batches), start_time)

    print()
    assert len(X) == len(y), "ASD"

    X = np.array(X)
    y = np.array(y).astype(np.int32)
    idxs = list(range(len(y)))
    np.random.shuffle(idxs)
    tr = int(0.9 * len(y))

    X_tr = X[:tr]
    y_tr = y[:tr]

    X_dev = X[tr:]
    y_dev = y[tr:]

    X_path = os.sep.join([VARS_DIR, "X_gloss_"])
    y_path = os.sep.join([VARS_DIR, "y_gloss_"])
    if not with_blank:
        X_path += "no_blank_"
        y_path += "no_blank_"

    np.save(X_path + "train", X_tr)
    np.save(y_path + "train", y_tr)
    np.save(X_path + "dev", X_dev)
    np.save(y_path + "dev", y_dev)

    print(X_tr.shape, y_tr.shape)
    print(X_dev.shape, y_dev.shape)


if __name__ == "__main__":
    # extract_features()

    generate_gloss_dataset(with_blank=False)
