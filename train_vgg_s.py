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
from dataset import get_batches_vgg_s_pheonix, get_tensor_batch_vgg_s
from train import predict_glosses
from utils import Vocab, ProgressPrinter
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


def get_split_wer(model, device, X_batches, y_batches, vocab, beam_search=False):
    hypes = []
    gts = []
    model.eval()

    decoder = None
    if beam_search:
        decoder = ctcdecode.CTCBeamDecoder(vocab.idx2gloss, beam_width=20,
                                           blank_id=0, log_probs_input=True)

    with torch.no_grad():

        for idx in range(len(X_batches)):
            X_batch, y_batch = X_batches[idx], y_batches[idx]

            inp = get_tensor_batch_vgg_s(X_batch).to(device)
            preds = model(inp).permute(1, 0, 2)
            out, out_idx = predict_glosses(preds, vocab, decoder)

            gt_batch = [" ".join(gt) for gt in vocab.decode_batch(y_batch)]

            if idx == 10:
                print("EXAMPLE: [" + gt_batch[0] + "]", "[" + out[0] + "]")

            for gt in y_batch:
                gts += gt

            hypes += out_idx

    hypes = "".join([chr(x) for x in hypes])
    gts = "".join([chr(x) for x in gts])
    return Lev.distance(hypes, gts) / len(gts) * 100


def train(model, device, vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, optimizer, n_epochs, batch_size):
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=5)

    loss_fn = nn.CTCLoss(zero_infinity=True)
    best_train_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)

        tr_losses = []

        model.train()
        pp = ProgressPrinter(len(X_tr), 10)
        for idx in range(len(X_tr)):
            optimizer.zero_grad()

            X_batch, y_batch = X_tr[idx], y_tr[idx]

            inp = get_tensor_batch_vgg_s(X_batch).to(device)
            pred = model(inp).log_softmax(dim=2)

            T, N, V = pred.shape

            gt = torch.IntTensor(y_batch[0])
            inp_lens = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            gt_lens = torch.IntTensor(y_batch[1])

            loss = loss_fn(pred, gt, inp_lens, gt_lens)

            if torch.isnan(loss):
                print("NAN!!")

            tr_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            pp.show(idx)

        print()

        train_loss = np.mean(tr_losses)
        print("Train Loss:", train_loss)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr_vgg_s.pt.pt"]))
            print("Model Saved")

        # if scheduler:
        #     scheduler.step(dev_wer)

        print()
        print()


if __name__ == "__main__":
    vocab = Vocab(source="pheonix")
    X_tr, y_tr = get_batches_vgg_s_pheonix("train", vocab, max_batch_size=8, shuffle=True, target_format=1)
    X_test, y_test = get_batches_vgg_s_pheonix("test", vocab, max_batch_size=8, shuffle=False)
    X_dev, y_dev = get_batches_vgg_s_pheonix("dev", vocab, max_batch_size=8, shuffle=False)
    print()

    device = torch.device("cuda:0")
    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=2).to(device)
    load = False
    if load and os.path.exists(os.sep.join([WEIGHTS_DIR, "slr_vgg_s.pt"])):
        model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr_vgg_s.pt"])))
        print("Model Loaded")
    else:
        model.apply(weights_init)

    lr = 0.005
    # optimizer = SGD(model.parameters(), lr=lr, nesterov=True)
    # optimizer = RMSprop(model.parameters(), lr=lr)
    optimizer = Adam(model.parameters(), lr=lr)
    train(model, device, vocab, X_tr, y_tr, X_dev, y_dev, X_test, y_test, optimizer, n_epochs=100, batch_size=8)
