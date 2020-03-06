import time
import torch
# import ctcdecode
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy import random
from models import SLR, weights_init
from dataset import read_pheonix_cnn_feats, load_gloss_dataset, PhoenixFullFeatDataset

from utils import ProgressPrinter, split_batches, Vocab
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


def predict_glosses(preds, decoder):
    out_sentences = []
    if decoder:
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(preds)
        for i in range(preds.size(0)):
            hypo = list(beam_result[i][0][:out_seq_len[i][0]])
            out_sentences.append(hypo)

    else:
        preds = preds.argmax(dim=2).cpu().numpy()
        # glosses_batch = vocab.decode_batch(preds)
        for pred in preds:
            hypo = []
            for i in range(len(pred)):
                if pred[i] == 0 or (i > 0 and pred[i] == pred[i - 1]):
                    continue
                hypo.append(pred[i])

            out_sentences.append(hypo)

    return out_sentences


def get_split_wer(model, device, X, Y, vocab, batch_size=16, beam_search=False):
    hypes = []
    gts = []

    decoder = None
    if beam_search:
        decoder = ctcdecode.CTCBeamDecoder(vocab.idx2gloss, beam_width=20,
                                           blank_id=0, log_probs_input=True)

    with torch.no_grad():
        X_batches, Y_batches = split_batches(X, Y, batch_size, shuffle=False, target_format=2)

        for idx in range(len(X_batches)):
            X_batch, Y_batch = X_batches[idx], Y_batches[idx]
            inp = torch.Tensor(X_batch).unsqueeze(1).to(device)
            preds = model(inp).log_softmax(dim=2).permute(1, 0, 2)
            out_idx = predict_glosses(preds, decoder)

            for gt in Y_batch:
                gts += gt

            hypes += out_idx

    hypes = "".join([chr(x) for x in hypes])
    gts = "".join([chr(x) for x in gts])
    return Lev.distance(hypes, gts) / len(gts) * 100


def train(model, loaded, device, vocab, tr_dataset, val_dataset, n_epochs):
    optimizer = Adam(model.parameters(), lr=LR_FULL)

    datasets = {"Train": tr_dataset, "Val": val_dataset}
    loss_fn = nn.CTCLoss(zero_infinity=True)

    criterion_phase = CRIT_PHASE_END2END_FULL
    best_wer_dir = os.sep.join([VARS_DIR, "PheonixWER"])
    best_wer_file = os.sep.join([best_wer_dir, "END2END_FULL_" + criterion_phase + ".txt"])
    if os.path.exists(best_wer_file) and loaded:
        with open(best_wer_file, 'r') as f:
            best_wer = float(f.readline().strip())
            print("BEST " + criterion_phase + " WER:", best_wer)
    else:
        if not os.path.exists(best_wer_dir):
            os.makedirs(best_wer_dir)
        best_wer = float("inf")

    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            dataset = datasets[phase]
            n_batches = dataset.start_epoch()
            losses = []
            hypes = []
            gts = []

            with torch.set_grad_enabled(phase == "Train"):
                pp = ProgressPrinter(n_batches, 25)
                for i in range(n_batches):
                    optimizer.zero_grad()
                    X_batch, Y_batch, Y_lens = dataset.get_batch(i)
                    X_batch = torch.Tensor(X_batch).unsqueeze(1).to(device)

                    preds = model(X_batch).log_softmax(dim=2)

                    T, N, V = preds.shape
                    X_lens = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
                    loss = loss_fn(preds, Y_batch, X_lens, Y_lens)

                    if torch.isnan(loss):
                        print("NAN!!")

                    losses.append(loss.item())

                    if phase == "Train":
                        loss.backward()
                        optimizer.step()

                    out_sentences = predict_glosses(preds, decoder=None)
                    gts += [y for y in Y_batch.view(-1).tolist() if y != 0]

                    for sentence in out_sentences:
                        hypes += sentence

                    if i == 0:
                        pred = " ".join(vocab.decode(out_sentences[0]))
                        gt = Y_batch[0][:Y_lens[0]].tolist()
                        gt = " ".join(vocab.decode(gt))
                        print(phase, 'Ex. [' + pred + ']', '[' + gt + ']')

                    if SHOW_PROGRESS:
                        pp.show(i)

                if SHOW_PROGRESS:
                    pp.end()

            hypes = "".join([chr(x) for x in hypes])
            gts = "".join([chr(x) for x in gts])
            phase_wer = Lev.distance(hypes, gts) / len(gts) * 100

            phase_loss = np.mean(losses)
            print(phase, "WER:", phase_wer, "Loss:", phase_loss)

            if phase == criterion_phase and phase_wer < best_wer:
                best_wer = phase_wer
                with open(best_wer_file, 'w') as f:
                    f.write(str(best_wer) + "\n")

                torch.save(model.state_dict(), END2END_FULL_MODEL_PATH)
                print("Model Saved")

        print()
        print()


def train_temp_fusion(vocab, X_tr, Y_tr, X_dev, Y_dev, n_epochs=100, batch_size=8192, lr=0.001):
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

    Y_tr = torch.LongTensor(Y_tr).to(device)
    Y_dev = torch.LongTensor(Y_dev).to(device)
    X_dev = torch.Tensor(X_dev).to(device).unsqueeze(1)

    with torch.no_grad():
        pred = model(X_dev).permute([1, 0, 2]).squeeze()

        best_dev_loss = loss_fn(pred, Y_dev).item()
        print("DEV LOSS:", best_dev_loss)

    n_batches = len(X_tr) // batch_size + 1 * (len(X_tr) % batch_size != 0)
    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        model.train()
        tr_losses = []
        for idx in range(n_batches):
            optimizer.zero_grad()
            start = idx * batch_size
            end = min(start + batch_size, len(Y_tr))
            X_batch = torch.Tensor(X_tr[start:end]).to(device).unsqueeze(1)
            Y_batch = Y_tr[start:end]
            out = model(X_batch).permute([1, 0, 2]).squeeze()

            loss = loss_fn(out, Y_batch)
            tr_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print("\rTrain Loss: ", np.mean(tr_losses))

        model.eval()
        with torch.no_grad():
            pred = model(X_dev).permute([1, 0, 2]).squeeze()

            dev_loss = loss_fn(pred, Y_dev).item()

            print("DEV LOSS:", dev_loss)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr_temp_fusion.pt"]))
                print("Model Saved")

            if scheduler:
                scheduler.step(dev_loss)


if __name__ == "__main__":
    vocab = Vocab(source="pheonix")

    # X_tr, Y_tr, X_dev, Y_dev = load_gloss_dataset(with_blank=False)
    # train_temp_fusion(vocab, X_tr, Y_tr, X_dev, Y_dev)

    tr_dataset = PhoenixFullFeatDataset(vocab, "train", max_batch_size=END2END_FULL_BATCH_SIZE,
                                        augment_temp=AUG_HAND_TEMP)

    val_dataset = PhoenixFullFeatDataset(vocab, "dev", max_batch_size=END2END_FULL_BATCH_SIZE)

    device = torch.device(DEVICE)
    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=0).to(device)
    load = True
    if load and os.path.exists(END2END_FULL_MODEL_PATH):
        loaded = True
        model.load_state_dict(torch.load(END2END_FULL_MODEL_PATH, map_location=DEVICE))
        print("Model Loaded")
    else:
        loaded = False
        model.apply(weights_init)

    train(model, loaded, device, vocab, tr_dataset, val_dataset, n_epochs=END2END_HAND_N_EPOCHS)
