import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from jiwer import wer
from numpy import random
from models import SLR, weights_init
from data import read_pheonix
from utils import Vocab, print_progress
from config import *

random.seed(0)


# TODO
# batch train/val
# debug from CRNN
# different optimizer
# different feature extraction


def calculate_loss(model, device, loss_fn, X, y):
    inp = torch.Tensor(X).unsqueeze(0).unsqueeze(0)
    gt = torch.LongTensor(y)
    pred = model(inp).permute(1, 0, 2)
    inp_lens = torch.IntTensor([pred.size()[0]])
    gt_lens = torch.IntTensor([gt.size()[0]])
    return loss_fn(pred, gt, inp_lens, gt_lens), pred


def predict_glosses(model, device, X):
    inp = torch.Tensor(X).unsqueeze(1).to(device)
    pred = model(inp)
    out = pred.squeeze().cpu().numpy().argmax(axis=1)
    return pred


def get_split_wer(model, device, X, y, vocab, batch_size=16):
    hypes = []
    gts = []
    model.eval()
    with torch.no_grad():
        X_batches, y_batches = split_batches(X, y, batch_size, concat_targets=True)

        for idx in range(len(X_batches)):
            X_batch, y_batch = X_batches[idx], y_batches[idx]

            pred = predict_glosses(model, device, X_batch)

            gt = " ".join([x for x in vocab.decode(y[idx]) if x != "-"])

            glosses = vocab.decode(out)
            hyp = []
            for i in range(len(glosses)):
                if glosses[i] == '-' or (i > 0 and glosses[i] == glosses[i - 1]):
                    continue

                hyp.append(glosses[i])

            hypes.append(" ".join(hyp))
            gts.append(gt)

    return wer(gts, hypes)


def train(model, device, vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, optimizer, n_epochs, batch_size):
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=5)

    loss_fn = nn.CTCLoss(blank=0, reduction='none')

    best_dev_wer = get_split_wer(model, device, X_dev, y_dev, vocab)
    print("DEV WER:", best_dev_wer)
    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        start_time = time.time()
        tr_losses = []
        model.train()
        X_batches, y_batches = split_batches(X_tr, y_tr, batch_size, concat_targets=True)
        for idx in range(len(X_batches)):
            optimizer.zero_grad()
            X_batch, y_batch = X_batches[idx], y_batches[idx]

            loss, _ = calculate_loss(model, device, loss_fn, X_batch, y_batch)
            tr_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print_progress(idx + 1, len(y_tr), start_time)

        print()
        dev_wer = get_split_wer(model, device, loss_fn, X_dev, y_dev, vocab)

        if dev_wer < best_dev_wer:
            test_wer = get_split_wer(model, device, loss_fn, X_test, y_test, vocab)
            dev_wer = best_dev_wer
            torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr.pt"]))
            print("Model Saved", "TEST WER:", test_wer)

        if scheduler:
            scheduler.step(dev_wer)

        print("DEV WER:", dev_wer)
        print()
        print()


def split_batches(X, y, max_batch_size, shuffle=True, concat_targets=False):
    len_table = dict()

    for idx, feats in enumerate(X):
        l = len(feats)
        if l in len_table:
            len_table[l].append(idx)
        else:
            len_table[l] = [idx]

    X_batches = []
    y_batches = []

    lenghts = list(len_table)

    if shuffle:
        random.shuffle(lenghts)

    for l in lenghts:
        idxs = len_table[l]
        if shuffle:
            random.shuffle(idxs)

        s = 0
        while (s < len(idxs)):
            e = s + max_batch_size
            if e > len(idxs):
                e = len(idxs)

            X_batches.append([X[i] for i in idxs[s:e]])
            if concat_targets:
                # target lengths too
                y_batch = []
                y_lengths = []
                for i in idxs[s:e]:
                    y_batch += y[i]
                    y_lengths.append(len(y[i]))

                y_batch = (y_batch, y_lengths)
            else:
                y_batch = [y[i] for i in idxs[s:e]]
            y_batches.append(y_batch)

            s += max_batch_size

    return X_batches, y_batches


if __name__ == "__main__":
    vocab = Vocab(source="pheonix")
    X_tr, y_tr = read_pheonix("train", vocab, save=True)
    X_test, y_test = read_pheonix("test", vocab, save=True)
    X_dev, y_dev = read_pheonix("dev", vocab, save=True)
    print()

    device = torch.device("cuda:0")
    model = SLR(rnn_hidden=512, vocab_size=vocab.size).to(device)

    if os.path.exists(os.sep.join([WEIGHTS_DIR, "slr.pt"])):
        model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr.pt"])))
        print("Model Loaded")
    else:
        model.apply(weights_init)

    lr = 0.01
    weight_decay = 1e-5
    momentum = 0.9
    optimizer = SGD(model.parameters(), lr=lr, nesterov=True,
                    weight_decay=weight_decay, momentum=momentum)

    train(model, device, vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, optimizer, n_epochs=10, batch_size=8)
