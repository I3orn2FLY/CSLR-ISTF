import pandas as pd
from utils import *
from config import *
import numpy as np
import pickle


def get_pheonix_df(split):
    path = os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"])
    return pd.read_csv(path, sep='|')


def pad_image(feats, VIDEO_SEQ_LEN):
    padded_feats = np.zeros((VIDEO_SEQ_LEN, feats.shape[1]))
    L = feats.shape[0]
    if L > VIDEO_SEQ_LEN:
        step = L // VIDEO_SEQ_LEN
        left_over = L % VIDEO_SEQ_LEN
        start_idx = step // 2
        start_idxs = []

        for i in range(VIDEO_SEQ_LEN):
            start_idxs.append(start_idx)

            padded_feats[i] = feats[start_idx]
            if np.random.rand() < left_over / VIDEO_SEQ_LEN and L - 1 > start_idx + step * (VIDEO_SEQ_LEN - i - 1):
                start_idx += step + 1
            else:
                start_idx += step
    else:
        for i in range(VIDEO_SEQ_LEN):
            if i < L:
                padded_feats[i] = feats[i]
            else:
                padded_feats[i] = feats[L - 1]

    return padded_feats


def read_pheonix(split, vocab, save=False):
    if VIDEO_SEQ_LEN:
        suffix = "_" + split + "_" + str(VIDEO_SEQ_LEN) + ".pkl"
    else:
        suffix = "_" + split + ".pkl"

    X_path = os.sep.join([VARS_DIR, 'X' + suffix])

    y_path = os.sep.join([VARS_DIR, 'y' + suffix])

    if os.path.exists(X_path) and os.path.exists(y_path):
        with open(X_path, 'rb') as f:
            X = pickle.load(f)

        with open(y_path, 'rb') as f:
            y = pickle.load(f)

        return X, y

    df = get_pheonix_df(split)
    X = []
    y = []
    print("Reading", split, "split")
    start_time = time.time()
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        text = row.annotation
        feat_path = os.sep.join([VIDEO_FEAT_DIR, split, row.folder]).replace("/*.png", ".npy")
        feats = np.load(feat_path)
        if VIDEO_SEQ_LEN:
            feats = pad_image(feats, VIDEO_SEQ_LEN)

        vectors = vocab.encode(text)
        if VIDEO_SEQ_LEN:
            vectors = vectors[:MAX_OUT_LEN]

        X.append(feats)
        y.append(vectors)

        if idx % 25:
            print_progress(idx + 1, df.shape[0], start_time)

    print()

    if save:
        with open(X_path, 'wb') as f:
            pickle.dump(X, f)

        with open(y_path, 'wb') as f:
            pickle.dump(y, f)

    return X, y


if __name__ == "__main__":
    vocab = Vocab()
    X_tr, y_tr = read_pheonix("train", vocab)
    df = get_pheonix_df("train")
    for i in range(len(X_tr)):
        if X_tr[i].shape[0] // 4 < y_tr[i].shape[0]:
            print(df.iloc[i].annotation)
            # print(feat_len)
            print(df.iloc[i].folder)

    print(len(X_tr), len(y_tr))
