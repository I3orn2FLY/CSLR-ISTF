import torch
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from torch.optim import Adam
import sys
import os
import pickle

sys.path.append(".." + os.sep)
from utils import ProgressPrinter, Vocab
from common import predict_glosses
from dataset import get_end2end_datasets
from models import SLR
from config import *

np.random.seed(0)

torch.backends.cudnn.deterministic = True


def get_wer_info(phases=["Train", "Val"]):
    best_wer = {phase: float("inf") for phase in phases}

    for phase in phases:
        wer_path = phase_path(END2END_WER_PATH, phase)
        if os.path.exists(wer_path):
            with open(wer_path, 'r') as f:
                best_wer[phase] = float(f.readline().strip())

        print("BEST", phase, "WER:", best_wer[phase])

    return best_wer


def phase_path(name, phase):
    ext = os.path.splitext(name)[1]
    if phase == "Train":
        return name.replace(ext, "_Train" + ext)
    return name


def save_model(model, phase, best_wer):
    model_dir = os.path.split(END2END_MODEL_PATH)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_wer_dir = os.path.split(END2END_WER_PATH)[0]
    if not os.path.exists(best_wer_dir):
        os.makedirs(best_wer_dir)

    with open(phase_path(END2END_WER_PATH, phase), 'w') as f:
        f.write(str(best_wer) + "\n")

    torch.save(model.state_dict(), phase_path(END2END_MODEL_PATH, phase))
    print("Model Saved")


def get_end2end_model(vocab):
    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=TEMP_FUSION_TYPE).to(DEVICE)

    if END2END_MODEL_LOAD:
        model_path = phase_path(END2END_MODEL_PATH, "Train") if USE_OVERFIT else END2END_MODEL_PATH

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("Model Loaded")
        else:
            print("Model Initialized")

    return model


def train(model, vocab, datasets):
    print("END2END model training...")
    print("Mode:", END2END_TRAIN_MODE)
    print("Features:", IMG_FEAT_MODEL)
    print("Save Model path:", END2END_MODEL_PATH)
    print("WER path:", END2END_WER_PATH)

    optimizer = Adam(model.parameters(), lr=END2END_LR)
    loss_fn = nn.CTCLoss(zero_infinity=True)

    best_wer = get_wer_info()
    curve = {"Train": [], "Val": []}
    try:

        for epoch in range(1, END2END_N_EPOCHS + 1):
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
                        X_batch = X_batch.to(DEVICE)

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

                        if i == 0 and SHOW_EXAMPLE:
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

                curve[phase].append(phase_wer)
                phase_loss = np.mean(losses)
                print(phase, "WER:", phase_wer, "Loss:", phase_loss)

                if phase_wer < best_wer[phase]:
                    best_wer[phase] = phase_wer
                    save_model(model, phase, best_wer[phase])

            print()
            print()
    except KeyboardInterrupt:
        pass
    print("\nTraining complete:", "Best WER:", best_wer)

    with open(os.path.join(VARS_DIR, "curve.pkl"), 'wb') as f:
        pickle.dump(curve, f)


if __name__ == "__main__":
    vocab = Vocab()

    model = get_end2end_model(vocab)
    datasets = get_end2end_datasets(vocab)

    train(model, vocab, datasets)
