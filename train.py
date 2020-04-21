# import ctcdecode
import torch
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from numpy import random
from torch.optim import Adam, RMSprop, SGD
from utils import ProgressPrinter, Vocab, predict_glosses
from dataset import get_end2end_datasets
from models import SLR, weights_init
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


def get_wer_info(loaded):
    best_wer_dir = os.path.split(END2END_WER_PATH)[0]
    if not os.path.exists(best_wer_dir):
        os.makedirs(best_wer_dir)

    if os.path.exists(END2END_WER_PATH) and loaded:
        with open(END2END_WER_PATH, 'r') as f:
            best_wer = float(f.readline().strip())
            print("BEST " + END2END_CRIT_PHASE + " WER:", best_wer)
    else:
        best_wer = float("inf")
        print("BEST " + END2END_CRIT_PHASE + " WER:", "Not found")

    return best_wer


def save_model(model, best_wer):
    with open(END2END_WER_PATH, 'w') as f:
        f.write(str(best_wer) + "\n")

    model_dir = os.path.split(END2END_MODEL_PATH)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), END2END_MODEL_PATH)
    print("Model Saved")


def get_end2end_model(vocab, load=True):
    if END2END_TRAIN_MODE == "FULL":
        temp_fusion_type = 0
    elif END2END_TRAIN_MODE == "HAND":
        temp_fusion_type = 1

    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=temp_fusion_type).to(DEVICE)
    if load and os.path.exists(END2END_MODEL_PATH):
        loaded = True
        model.load_state_dict(torch.load(END2END_MODEL_PATH, map_location=DEVICE))
        print("Model Loaded")
    else:
        loaded = False
        model.apply(weights_init)

    return model, loaded


def train(model, loaded, vocab, datasets):
    print("END2END model training...")
    print("Mode:", END2END_TRAIN_MODE)
    print("Features:", FRAME_FEAT_MODEL)
    print("Model path:", END2END_MODEL_PATH)
    print("WER path:", END2END_WER_PATH)

    if END2END_TRAIN_OPTIMIZER == "Adam":
        optimizer = Adam(model.parameters(), lr=END2END_LR)
    elif END2END_TRAIN_OPTIMIZER == "RMSProp":
        optimizer = RMSprop(model.parameters(), lr=END2END_LR)
    else:
        optimizer = SGD(model.parameters(), lr=END2END_LR)

    loss_fn = nn.CTCLoss(zero_infinity=True)

    best_wer = get_wer_info(loaded)
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
                        L = X_batch.size()[-2]
                        if L < 4 * Y_lens.max().item():
                            batch = dataset.batches[i]
                            paths = [dataset.X[k] for k in batch]
                            print(paths)
                            continue

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

                phase_loss = np.mean(losses)
                print(phase, "WER:", phase_wer, "Loss:", phase_loss)

                if phase == END2END_CRIT_PHASE and phase_wer < best_wer:
                    best_wer = phase_wer
                    save_model(model, best_wer)

            print()
            print()
    except KeyboardInterrupt:
        pass
    print("\nTraining complete:", "Best WER:", best_wer)


if __name__ == "__main__":
    vocab = Vocab()

    model, loaded = get_end2end_model(vocab)
    datasets = get_end2end_datasets(vocab)

    train(model, loaded, vocab, datasets)
