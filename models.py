import torch
import torch.nn as nn
import torchvision.models as models
from config import *


class Identity(nn.Module):
    def forward(self, x):
        return x


class FrameFeatModel(nn.Module):
    def __init__(self):
        super(FrameFeatModel, self).__init__()
        if FRAME_FEAT_MODEL == "densenet121":
            self.ffm = models.densenet121(pretrained=True)
        else:
            self.ffm = models.densenet121(pretrained=True)

        self.ffm.classifier = Identity()

    def forward(self, x):
        return self.ffm(x)


class TempFusion(nn.Module):
    def __init__(self):
        super(TempFusion, self).__init__()

        self.conv1d_1 = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv1d_2 = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.pool1(x)
        x = self.conv1d_2(x)
        x = self.pool2(x)
        return x.squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=FRAME_FEAT_SIZE, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.emb = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.emb(x)
        return x


class SLR(nn.Module):

    def __init__(self, rnn_hidden, vocab_size):
        super(SLR, self).__init__()
        self.temp_fusion = TempFusion()
        self.seq_model = BiLSTM(rnn_hidden, vocab_size)

    def forward(self, x):
        x = self.temp_fusion(x)
        x = self.seq_model(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    tmp = TempFusion()
    split = "train"
    df = pd.read_csv(os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"]), sep='|')

    row = df.iloc[500]
    feat_dir = os.sep.join([VIDEO_FEAT_DIR, split, row.folder])
    feat_file = feat_dir.replace("/*.png", ".npy")

    feats = np.load(feat_file)
    inp = torch.stack([torch.Tensor(feats).unsqueeze(0), torch.Tensor(feats).unsqueeze(0)])
    # inp = torch.rand([2, 1, 27, 1024])

    slr = SLR(rnn_hidden=512)

    out = slr(inp)
    print(inp.shape)
    print(out.shape)
