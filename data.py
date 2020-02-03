import pandas as pd
from utils import *
from config import *
import pickle


def get_pheonix_df(split):
    path = os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"])
    return pd.read_csv(path, sep='|')


def read_pheonix(split, vocab, tensor=False, save=False):
    if os.path.exists(os.sep.join([VARS_DIR, 'X_' + split + '.pkl'])) \
            and os.path.exists(os.sep.join([VARS_DIR, 'y_' + split + '.pkl'])):
        with open(os.sep.join([VARS_DIR, 'X_' + split + '.pkl']), 'rb') as f:
            X = pickle.load(f)

        with open(os.sep.join([VARS_DIR, 'y_' + split + '.pkl']), 'rb') as f:
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
        vectors = vocab.encode(text)

        if feats.shape[0] // 4 < len(vectors):
            continue
        X.append(feats)
        y.append(vectors)

        if idx % 25:
            print_progress(idx + 1, df.shape[0], start_time)

    print()

    if save:
        with open(os.sep.join([VARS_DIR, 'X_' + split + '.pkl']), 'wb') as f:
            pickle.dump(X, f)

        with open(os.sep.join([VARS_DIR, 'y_' + split + '.pkl']), 'wb') as f:
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
