import torch
import torch.nn as nn
import numpy as np
from numpy import random
from torch.optim import Adam

from utils import ProgressPrinter, Vocab
from dataset import get_gr_datasets
from models import get_GR_model
from config import *

random.seed(0)
torch.backends.cudnn.enabled = False


def get_best_loss():
    if os.path.exists(GR_LOSS_PATH):
        with open(GR_LOSS_PATH, 'r') as f:
            best_loss = float(f.readline().strip())
            print("BEST GR LOSS:", best_loss)
    else:
        best_loss = float("inf")
        print("BEST GR LOSS:", best_loss)

    return best_loss


def save_model(model, best_loss):
    best_loss_dir = os.path.split(GR_LOSS_PATH)[0]
    if not os.path.exists(best_loss_dir):
        os.makedirs(best_loss_dir)

    with open(GR_LOSS_PATH, 'w') as f:
        f.write(str(best_loss) + "\n")

    torch.save(model.stf.state_dict(), STF_MODEL_PATH)
    print("    ", "Model Saved")


def train_gloss_recog(model, datasets):
    print("GR model training...")
    print("Mode:", SRC_MODE)
    print("Features:", STF_MODEL)
    best_loss = float("inf")
    optimizer = Adam(model.parameters(), lr=GR_LR)

    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    trained = False

    # n_epochs since wer was updated
    for epoch in range(1, GR_N_EPOCHS + 1):
        print("Epoch", epoch)
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            dataset = datasets[phase]
            n_batches = dataset.start_epoch()
            losses = []

            correct = []

            with torch.set_grad_enabled(phase == "Train"):
                pp = ProgressPrinter(n_batches, 25)
                for i in range(n_batches):
                    if phase == "Train":
                        optimizer.zero_grad()

                    X_batch, Y_batch = dataset.get_batch(i)

                    X_batch = X_batch.to(DEVICE)
                    Y_batch = Y_batch.to(DEVICE)

                    preds = model(X_batch)
                    loss = loss_fn(preds, Y_batch)

                    correct.append(torch.sum(preds.argmax(dim=1) == Y_batch).item())

                    losses.append(loss.item())

                    if phase == "Train":
                        loss.backward()
                        optimizer.step()

                    if SHOW_PROGRESS:
                        pp.show(i, "    Loss: %.3f" % np.mean(losses))

                if SHOW_PROGRESS:
                    pp.end("    ")

            phase_loss = np.mean(losses)
            phase_acc = sum(correct) / len(correct * GR_BATCH_SIZE) * 100

            print("    ", phase, "loss:", phase_loss, "phase ACC:", phase_acc)

            if phase == "Val" and phase_loss < best_loss:
                best_loss = phase_loss
                save_model(model, best_loss)

            if phase == "Val":
                best_acc = max(best_acc, phase_acc)

        if epoch >= 5:
            trained = True

    return best_acc, trained


if __name__ == "__main__":
    vocab = Vocab()
    model = get_GR_model(vocab)
    datasets = get_gr_datasets()
    best_acc, trained = train_gloss_recog(model, datasets)
    print("\nTraining complete:", "Best ACC:", best_acc, "Finished:", trained)
