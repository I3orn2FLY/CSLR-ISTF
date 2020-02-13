import time
import torch
import ctcdecode
import torch.nn as nn
import numpy as np
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from jiwer import wer
from numpy import random
from models import SLR, weights_init
from data import read_pheonix
from utils import Vocab, print_progress
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


# TODO
# debug from CRNN
# try tensorflow version ctcloss
# try warp-ctc with pytorch 0.4
# cnn-hmm

# permutations
# different feature extraction

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


def predict_glosses(model, device, X, vocab, decoder):
    # Takes only batches now
    inp = torch.Tensor(X).unsqueeze(1).to(device)
    preds = model(inp).log_softmax(dim=2)
    out = []
    preds = preds.argmax(dim=2).cpu().numpy()

    glosses_batch = vocab.decode_batch(preds)
    for glosses in glosses_batch:
        hypo = []
        for idx, gloss in enumerate(glosses):
            if gloss == '-' or (idx > 0 and glosses[idx] == glosses[idx - 1]):
                continue
            hypo.append(gloss)

        out.append(" ".join(hypo))

    # beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(preds)
    # for i in range(preds.size(0)):
    #     hypo = beam_result[i][0][:out_seq_len[i][0]]
    #
    #     out.append(" ".join([vocab.idx2gloss[x] for x in hypo]))

    return out


def get_split_wer(model, device, X, y, vocab, batch_size=16):
    hypes = []
    gts = []
    model.eval()
    decoder = ctcdecode.CTCBeamDecoder(vocab.idx2gloss, beam_width=20,
                                       blank_id=0, log_probs_input=True)

    with torch.no_grad():
        X_batches, y_batches = split_batches(X, y, batch_size, concat_targets=False, shuffle=False)

        for idx in range(len(X_batches)):
            X_batch, y_batch = X_batches[idx], y_batches[idx]

            out = predict_glosses(model, device, X_batch, vocab, decoder)

            gt_batch = [" ".join(gt) for gt in vocab.decode_batch(y_batch)]

            gts += gt_batch

            hypes += out

    print("EXAMPLE: [" + gts[200] + "]", "[" + hypes[200] + "]")
    return wer(gts, hypes)


def train(model, device, vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, optimizer, n_epochs, batch_size):
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=5)

    loss_fn = nn.CTCLoss()

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

            inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
            pred = model(inp).permute(1, 0, 2).log_softmax(dim=2)

            T, N, V = pred.shape

            gt = torch.IntTensor(y_batch[0])
            inp_lens = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            gt_lens = torch.IntTensor(y_batch[1])

            loss = loss_fn(pred, gt, inp_lens, gt_lens)

            # loss_batch = loss.detach().cpu().numpy()
            # tr_losses += [x for x in loss_batch]

            tr_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print_progress(idx + 1, len(y_batches), start_time)

        print()
        print("Train Loss:", np.mean(tr_losses))

        dev_wer = get_split_wer(model, device, X_dev, y_dev, vocab)
        print("DEV WER:", dev_wer)
        if dev_wer < best_dev_wer:
            # test_wer = get_split_wer(model, device, X_test, y_test, vocab)
            test_wer = dev_wer
            best_dev_wer = dev_wer
            torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr.pt"]))
            print("Model Saved", "TEST WER:", test_wer)

        # if scheduler:
        #     scheduler.step(dev_wer)

        print()
        print()


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

    lr = 0.0001
    # optimizer = SGD(model.parameters(), lr=lr, nesterov=True)
    optimizer = RMSprop(model.parameters(), lr=lr)
    # optimizer = Adam(model.parameters(), lr=lr)
    train(model, device, vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, optimizer, n_epochs=100, batch_size=8)

    # y_tr
