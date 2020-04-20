# import ctcdecode
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from numpy import random

from train_utils import *
from utils import ProgressPrinter, Vocab
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


def train(mode, model, loaded, model_path, vocab, datasets):
    optimizer, n_epochs = get_train_info(mode, model)

    loss_fn = nn.CTCLoss(zero_infinity=True)

    best_wer, best_wer_file, criterion_phase = get_wer_info(mode, loaded)
    try:
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
                    save_model(model, model_path, best_wer, best_wer_file)

            print()
            print()
    except KeyboardInterrupt:
        pass
    print("Training complete:", "Best WER:", best_wer)


if __name__ == "__main__":
    vocab = Vocab()

    model, loaded, model_path = get_model(END2END_TRAIN_MODE, vocab, END2END_TRAIN_LOAD)
    datasets = get_datasets(END2END_TRAIN_MODE, vocab)

    train(END2END_TRAIN_MODE, model, loaded, model_path, vocab, datasets)
