import torch
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
import pickle
from torch.optim import Adam

from utils import ProgressPrinter, Vocab
from common import predict_glosses
from dataset import get_end2end_datasets
from models import get_end2end_model
from config import *

np.random.seed(0)


# torch.backends.cudnn.deterministic = True


def get_best_wer():
    phases = ["train", "val"]
    best_wer = {phase: float("inf") for phase in phases}

    for phase in phases:
        wer_path = phase_path(END2END_WER_PATH, phase)
        if os.path.exists(wer_path):
            with open(wer_path, 'r') as f:
                best_wer[phase] = float(f.readline().strip())

    print("BEST WER:", best_wer)
    return best_wer


def phase_path(path, phase):
    dir, filename = os.path.split(path)
    filename = filename.replace("train", phase)
    filename = filename.replace("val", phase)
    return os.path.join(dir, filename)


def save_end2end_model(model, phase, best_wer, use_feat):
    model_dir = os.path.split(STF_MODEL_PATH)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_dir = os.path.split(STF_MODEL_PATH)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_wer_dir = os.path.split(END2END_WER_PATH)[0]
    if not os.path.exists(best_wer_dir):
        os.makedirs(best_wer_dir)

    with open(phase_path(END2END_WER_PATH, phase), 'w') as f:
        f.write(str(best_wer) + "\n")

    if not use_feat:
        torch.save(model.stf.state_dict(), phase_path(STF_MODEL_PATH, phase))
    torch.save(model.seq2seq.state_dict(), phase_path(SEQ2SEQ_MODEL_PATH, phase))
    print("   ", "Model Saved")


def train_end2end(model, vocab, datasets, use_feat):
    print("END2END model training...")
    print("Mode:", SRC_MODE)
    print("Features:", STF_MODEL)
    print("Save Model path:", STF_MODEL_PATH)
    print("WER path:", END2END_WER_PATH)

    optimizer = Adam(model.parameters(), lr=END2END_LR)
    loss_fn = nn.CTCLoss(zero_infinity=True)

    best_wer = get_best_wer()
    curve = {"train": [], "val": []}

    trained = False
    try:
        # n_epochs since wer was updated
        since_wer_update = -5
        for epoch in range(1, END2END_N_EPOCHS + 1):
            print("Epoch", epoch)
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()

                dataset = datasets[phase]
                n_batches = dataset.start_epoch()
                losses = []
                hypes = []
                gts = []

                with torch.set_grad_enabled(phase == "train"):
                    pp = ProgressPrinter(n_batches, 25 if USE_STF_FEAT else 1)
                    for i in range(n_batches):
                        optimizer.zero_grad()
                        X_batch, Y_batch, Y_lens = dataset.get_batch(i)
                        X_batch = X_batch.to(DEVICE)

                        preds = model(X_batch).log_softmax(dim=2)
                        T, N, V = preds.shape
                        X_lens = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
                        loss = loss_fn(preds, Y_batch, X_lens, Y_lens)
                        losses.append(loss.item())

                        if phase == "train":
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
                            print("   ", phase, 'Ex. [' + pred + ']', '[' + gt + ']')

                        if SHOW_PROGRESS:
                            pp.show(i, "   ")

                    if SHOW_PROGRESS:
                        pp.end("   ")

                hypes = "".join([chr(x) for x in hypes])
                gts = "".join([chr(x) for x in gts])
                phase_wer = Lev.distance(hypes, gts) / len(gts) * 100

                curve[phase].append(phase_wer)
                phase_loss = np.mean(losses)
                print("   ", phase.upper(), "WER:", phase_wer, "Loss:", phase_loss)

                if phase_wer < best_wer[phase]:
                    best_wer[phase] = phase_wer
                    save_end2end_model(model, phase, best_wer[phase], use_feat=use_feat)
                    if phase == "val":
                        since_wer_update = 0

                if phase == "val":
                    if since_wer_update >= END2END_STOP_LIMIT:
                        trained = True
                        raise KeyboardInterrupt
                    since_wer_update += 1

    except KeyboardInterrupt:
        pass

    with open(os.path.join(VARS_DIR, "curve.pkl"), 'wb') as f:
        pickle.dump(curve, f)

    return best_wer, trained


if __name__ == "__main__":
    vocab = Vocab()

    model, _ = get_end2end_model(vocab, END2END_MODEL_LOAD, END2END_MODEL_LOAD, STF_TYPE, USE_STF_FEAT)
    datasets = get_end2end_datasets(vocab)
    best_wer, trained = train_end2end(model, vocab, datasets, USE_STF_FEAT)

    print("\nEnd2End training complete:", "Best WER:", best_wer, "Finished:", trained)
