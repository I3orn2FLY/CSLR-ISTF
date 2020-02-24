import time
import torch
import ctcdecode
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy import random
from models import SLR, weights_init
from dataset import read_pheonix, load_gloss_dataset

from utils import ProgressPrinter, split_batches, Vocab
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


# TODO
# training with dataset/augmentation
# load to dgx
# may be try with densenet

# evaluate and try training samples,
# if they are ok, try fix lstm weights and re-train temporal fusion with predicted glosses


def predict_glosses(preds, decoder):
    out_sentences = []
    if decoder:
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(preds)
        for i in range(preds.size(0)):
            hypo = list(beam_result[i][0][:out_seq_len[i][0]])
            out_sentences += hypo

    else:
        preds = preds.argmax(dim=2).cpu().numpy()
        # glosses_batch = vocab.decode_batch(preds)
        for pred in preds:
            hypo = []
            for i in range(len(pred)):
                if pred[i] == 0 or (i > 0 and pred[i] == pred[i - 1]):
                    continue
                hypo.append(pred[i])

            out_sentences += hypo

    return out_sentences


def get_split_wer(model, device, X, y, vocab, batch_size=16, beam_search=False):
    hypes = []
    gts = []

    decoder = None
    if beam_search:
        decoder = ctcdecode.CTCBeamDecoder(vocab.idx2gloss, beam_width=20,
                                           blank_id=0, log_probs_input=True)

    with torch.no_grad():
        X_batches, y_batches = split_batches(X, y, batch_size, shuffle=False, target_format=2)

        for idx in range(len(X_batches)):
            X_batch, y_batch = X_batches[idx], y_batches[idx]
            inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
            preds = model(inp, 4).log_softmax(dim=2).permute(1, 0, 2)
            out_idx = predict_glosses(preds, decoder)

            for gt in y_batch:
                gts += gt

            hypes += out_idx

    hypes = "".join([chr(x) for x in hypes])
    gts = "".join([chr(x) for x in gts])
    return Lev.distance(hypes, gts) / len(gts) * 100


def train_end2end(vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, n_epochs, batch_size, lr=0.001, mode=0):
    device = DEVICE
    model = SLR(rnn_hidden=512, vocab_size=vocab.size).to(device)

    if os.path.exists(os.sep.join([WEIGHTS_DIR, "slr_temp_fusion.pt"])):
        model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr_temp_fusion.pt"])))
        print("Model Loaded")
    else:
        model.apply(weights_init)

    optimizer = Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    loss_fn = nn.CTCLoss(zero_infinity=True)

    # if mode == 0:
    #     objective = get_split_wer(model, device, X_dev, y_dev, vocab)
    #     print("DEV WER:", objective)
    # else:
    #     objective = float("inf")

    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        tr_losses = []
        model.train()
        X_batches, y_batches = split_batches(X_tr, y_tr, batch_size, target_format=1)

        pp = ProgressPrinter(len(y_batches), 10)
        for idx in range(len(X_batches)):
            optimizer.zero_grad()
            X_batch, y_batch = X_batches[idx], y_batches[idx]

            inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
            pred = model(inp, 4).log_softmax(dim=2)

            T, N, V = pred.shape
            gt = torch.IntTensor(y_batch[0])
            inp_lens = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            gt_lens = torch.IntTensor(y_batch[1])

            loss = loss_fn(pred, gt, inp_lens, gt_lens)
            tr_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            pp.show(idx)

        print()
        train_loss = np.mean(tr_losses)
        print("Train Loss:", train_loss)

        if mode == 0:
            model.eval()
            dev_wer = get_split_wer(model, device, X_dev, y_dev, vocab)
            print("DEV WER:", dev_wer)
            if dev_wer < objective:
                test_wer = get_split_wer(model, device, X_test, y_test, vocab)
                objective = dev_wer
                torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr.pt"]))
                print("Model Saved", "TEST WER:", test_wer)
                if scheduler:
                    scheduler.step(dev_wer)
        else:
            if train_loss < objective:
                objective = train_loss
                torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr.pt"]))
                print("Model Saved")

        print()
        print()


def train_temp_fusion(vocab, X_tr, y_tr, X_dev, y_dev, n_epochs=100, batch_size=8192, lr=0.001):
    print("Training temporal fusion model")
    device = DEVICE
    model = SLR(rnn_hidden=512, vocab_size=vocab.size).to(device)
    model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr_temp_fusion.pt"])))

    for param in model.seq_model.parameters():
        param.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.temp_fusion.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    scheduler = None

    y_tr = torch.LongTensor(y_tr).to(device)
    y_dev = torch.LongTensor(y_dev).to(device)
    X_dev = torch.Tensor(X_dev).to(device).unsqueeze(1)

    with torch.no_grad():
        pred = model(X_dev).permute([1, 0, 2]).squeeze()

        best_dev_loss = loss_fn(pred, y_dev).item()
        print("DEV LOSS:", best_dev_loss)

    n_batches = len(X_tr) // batch_size + 1 * (len(X_tr) % batch_size != 0)
    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        model.train()
        tr_losses = []
        for idx in range(n_batches):
            optimizer.zero_grad()
            start = idx * batch_size
            end = min(start + batch_size, len(y_tr))
            X_batch = torch.Tensor(X_tr[start:end]).to(device).unsqueeze(1)
            y_batch = y_tr[start:end]
            out = model(X_batch).permute([1, 0, 2]).squeeze()

            loss = loss_fn(out, y_batch)
            tr_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print("\rTrain Loss: ", np.mean(tr_losses))

        model.eval()
        with torch.no_grad():
            pred = model(X_dev).permute([1, 0, 2]).squeeze()

            dev_loss = loss_fn(pred, y_dev).item()

            print("DEV LOSS:", dev_loss)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr_temp_fusion.pt"]))
                print("Model Saved")

            if scheduler:
                scheduler.step(dev_loss)


if __name__ == "__main__":
    vocab = Vocab(source="pheonix")

    # X_tr, y_tr, X_dev, y_dev = load_gloss_dataset(with_blank=False)
    # train_temp_fusion(vocab, X_tr, y_tr, X_dev, y_dev)

    X_tr, y_tr = read_pheonix("train", vocab, save=True)
    X_test, y_test = read_pheonix("test", vocab, save=True)
    X_dev, y_dev = read_pheonix("dev", vocab, save=True)
    print()
    train_end2end(vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, n_epochs=200, batch_size=8, mode=0)
