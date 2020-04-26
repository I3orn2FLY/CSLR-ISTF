# import ctcdecode
import torch
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from numpy import random
from torch.optim import Adam, RMSprop, SGD
from utils import ProgressPrinter, Vocab, predict_glosses
from dataset import End2EndRawDataset
from models import SLR, weights_init
from config import *



def generate_gloss_dataset():
    vocab = Vocab()
    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=2).to(DEVICE)
    if os.path.exists(TEMP_FUSION_END2END_MODEL_PATH):
        model.load_state_dict(torch.load(TEMP_FUSION_END2END_MODEL_PATH, map_location=DEVICE))
        print("Model Loaded")
    else:
        print("Model doesnt exist")
        exit(0)


    model.eval()

    dataset = End2EndRawDataset(vocab, "train", 16, IMG_SIZE_3D, False, False)
    temp_stride = 4
    X = []
    y = []

    with torch.no_grad():
        n_batches = dataset.start_epoch()
        batches = dataset.batches
        pp = ProgressPrinter(n_batches, 5)
        gloss_idx = 0
        for idx in range(n_batches):
            batch_idxs = batches[idx]
            X_batch = dataset.get_X_batch(batch_idxs)
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).log_softmax(dim=2).permute(1, 0, 2)

            for i in range(preds.shape[0]):
                for j in range(len(preds[i])):
                    feat = X_batch[i, :, j * temp_stride: (j + 1) * temp_stride]
                    gloss = preds[i][j]
                    X.append(feat)
                    y.append(gloss)

            pp.show(idx)

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