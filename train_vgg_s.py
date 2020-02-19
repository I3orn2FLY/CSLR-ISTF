import torch
import torch.nn as nn
import numpy as np
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from numpy import random
from models import SLR, weights_init
from torch.utils.data import DataLoader
from dataset import PhoenixHandVideoDataset, hand_video_collate
from utils import Vocab, ProgressPrinter
from config import *

random.seed(0)

torch.backends.cudnn.deterministic = True


# add wer
# may be, investigate LSTM, masking

def train(model, device, vocab, tr_data_loader, val_data_loader, n_epochs):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x: x * 0.7)

    data_loaders = {"Train": tr_data_loader, "Val": val_data_loader}
    loss_fn = nn.CTCLoss(zero_infinity=True)
    best_val_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            losses = []
            pp = ProgressPrinter(len(data_loaders[phase]), 1)
            with torch.set_grad_enabled(phase == "Train"):
                for idx, (X_batch, x_lens, y_batch, y_lens) in enumerate(data_loaders[phase]):
                    optimizer.zero_grad()

                    X_batch, x_lens = X_batch.to(device), x_lens // 4
                    pred = model(X_batch, x_lens).log_softmax(dim=2)
                    loss = loss_fn(pred, y_batch, x_lens, y_lens)

                    if torch.isnan(loss):
                        print("NAN!!")

                    losses.append(loss.item())

                    if phase == "Train":
                        loss.backward()
                        optimizer.step()

                    pp.show(idx)

            pp.end()

            phase_loss = np.mean(losses)
            print(phase, "Loss:", phase_loss)

            if phase == "Val" and phase_loss < best_val_loss:
                best_val_loss = phase_loss
                torch.save(model.state_dict(), os.sep.join([WEIGHTS_DIR, "slr_vgg_s.pt.pt"]))
                print("Model Saved")

        if epoch % 10 == 0:
            scheduler.step()
        print()
        print()


if __name__ == "__main__":
    vocab = Vocab(source="pheonix")
    tr_dataset = PhoenixHandVideoDataset(vocab, "train", augment=True)
    val_dataset = PhoenixHandVideoDataset(vocab, "dev", augment=False)
    tr_data_loader = DataLoader(tr_dataset, batch_size=END2END_BATCH_SIZE, shuffle=True, collate_fn=hand_video_collate)
    val_data_loader = DataLoader(val_dataset, batch_size=END2END_BATCH_SIZE, shuffle=True,
                                 collate_fn=hand_video_collate)

    device = torch.device(DEVICE)
    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=2).to(device)
    load = False
    if load and os.path.exists(os.sep.join([WEIGHTS_DIR, "slr_vgg_s.pt"])):
        model.load_state_dict(torch.load(os.sep.join([WEIGHTS_DIR, "slr_vgg_s.pt"])))
        print("Model Loaded")
    else:
        model.apply(weights_init)

    lr = 0.001

    train(model, device, vocab, tr_data_loader, val_data_loader, n_epochs=150)
