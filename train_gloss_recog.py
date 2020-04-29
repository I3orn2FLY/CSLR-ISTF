# import ctcdecode
import torch
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
from numpy import random
from torch.optim import Adam
from utils import ProgressPrinter, Vocab
from dataset import get_gr_datasets
from models import SLR, GR
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


def get_best_acc():
    if os.path.exists(GR_ACC_PATH):
        with open(GR_ACC_PATH, 'r') as f:
            best_acc = float(f.readline().strip())
            print("BEST GR ACC:", best_acc)
    else:
        best_acc = 0
        print("BEST GR ACC:", best_acc)

    return best_acc


def split_model(vocab):
    if os.path.exists(GR_TF_MODEL_PATH) and not SPLIT_MODEL:
        return

    end2end_model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=1, use_feat=False).to(DEVICE)
    if os.path.exists(GR_END2END_MODEL_PATH):
        end2end_model.load_state_dict(torch.load(GR_END2END_MODEL_PATH, map_location=DEVICE))
        print("Model Loaded")
    else:
        print("Model doesnt exist")
        exit(0)

    gr_dir = os.path.split(GR_TF_MODEL_PATH)[0]
    if not os.path.exists(gr_dir):
        os.makedirs(gr_dir)

    temp_fusion = end2end_model.temp_fusion
    seq_model = end2end_model.seq_model

    torch.save(temp_fusion.state_dict(), GR_TF_MODEL_PATH)
    torch.save(seq_model.state_dict(), GR_SEQ_MODEL_PATH)


def get_GR_model(vocab):
    model = GR(vocab.size).to(DEVICE)
    if os.path.exists(GR_TF_MODEL_PATH):
        model.temp_fusion.load_state_dict(torch.load(GR_TF_MODEL_PATH, map_location=DEVICE))
        print("Temp fusion model Loaded")
    else:
        print("Temp fusion model doesnt exist")
        exit(0)

    if os.path.exists(GR_FC_PATH):
        model.fc.load_state_dict(torch.load(GR_FC_PATH, map_location=DEVICE))
        print("GR fc model Loaded")
    else:
        print("GR fc model doesnt exist")

    return model


def save_model(model, best_acc):
    best_acc_dir = os.path.split(GR_ACC_PATH)[0]
    if not os.path.exists(best_acc_dir):
        os.makedirs(best_acc_dir)

    with open(GR_ACC_PATH, 'w') as f:
        f.write(str(best_acc) + "\n")

    temp_fusion = model.temp_fusion
    fc_model = model.fc

    torch.save(temp_fusion.state_dict(), GR_TF_MODEL_PATH)
    torch.save(fc_model.state_dict(), GR_FC_PATH)
    print("Model Saved")


def train(model, datasets):
    print("GR model training...")
    print("Mode:", END2END_TRAIN_MODE)
    print("Features:", IMG_FEAT_MODEL)
    best_acc = get_best_acc()
    optimizer = Adam(model.parameters(), lr=GR_LR)

    loss_fn = nn.CrossEntropyLoss()

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

                        if torch.isnan(loss):
                            print("NAN!!")

                        losses.append(loss.item())

                        if phase == "Train":
                            loss.backward()
                            optimizer.step()

                        if SHOW_PROGRESS:
                            pp.show(i)

                    if SHOW_PROGRESS:
                        pp.end()

                phase_acc = sum(correct) / len(correct * GR_BATCH_SIZE) * 100

                print(phase, "phase ACC:", phase_acc, "loss:", np.mean(losses))

                if phase == "Val" and phase_acc > best_acc:
                    best_acc = phase_acc
                    save_model(model, best_acc)

            print()
            print()
    except KeyboardInterrupt:
        pass
    print("\nTraining complete:", "Best ACC:", best_acc)


if __name__ == "__main__":
    vocab = Vocab()
    split_model(vocab)
    model = get_GR_model(vocab)
    datasets = get_gr_datasets()
    train(model, datasets)