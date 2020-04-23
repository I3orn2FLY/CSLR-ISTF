import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import PIL
from config import *


class Identity(nn.Module):
    def forward(self, x):
        return x


class FrameFeatModel(nn.Module):
    def __init__(self):
        super(FrameFeatModel, self).__init__()
        if FRAME_FEAT_MODEL.startswith("densenet121"):
            self.ffm = models.densenet121(pretrained=True)
            self.ffm.classifier = Identity()
        elif FRAME_FEAT_MODEL.startswith("googlenet"):
            self.ffm = models.googlenet(pretrained=True)
            self.ffm.fc = Identity()
        elif FRAME_FEAT_MODEL.startswith("resnet18"):
            self.ffm = models.resnet18(pretrained=True)
            self.ffm.fc = nn.Identity()
        elif FRAME_FEAT_MODEL.startswith("vgg-s"):
            self.ffm = VGG_S

    def forward(self, x):
        return self.ffm(x)


class VGG_S(nn.Module):
    def __init__(self):
        super(VGG_S, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.lrn1 = nn.LocalResponseNorm(1)

        self.conv2 = nn.Conv2d(96, 256, 5, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)

        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.fc6 = nn.Linear(512 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.lrn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool5(x)

        x = x.view(-1, 512 * 2 * 2)

        x = self.fc6(x)
        x = F.relu(x)

        return x


class TempFusion(nn.Module):
    def __init__(self):
        super(TempFusion, self).__init__()

        self.conv1d_1 = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv1d_2 = nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        # (batch_size, 1, max_seq_length, 1024 or 411)
        x = self.conv1d_1(x)
        x = self.pool1(x)
        x = self.conv1d_2(x)
        x = self.pool2(x)
        x = x.squeeze(1)
        # (batch_size, max_seq_length // 4, 1024 or 411)
        return x


class TempFusion_FE(nn.Module):
    def __init__(self,):
        super(TempFusion_FE, self).__init__()

        self.fe = FrameFeatModel()
        if not END2END_TRAIN_FE:
            self.fe.eval()
            for param in self.fe.parameters():
                param.requires_grad = False

        self.simple_temp_fusion = TempFusion()

    def forward(self, x):
        # (batch_size, max_seq_length, 3, 101, 101)
        batch_feats = []
        for video_idx in range(x.shape[0]):
            video_feat = self.fe(x[video_idx])
            batch_feats.append(video_feat)
        batch_feats = torch.stack(batch_feats)

        x = batch_feats.unsqueeze(1)
        x = self.simple_temp_fusion(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=FRAME_FEAT_SIZE, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True)
        self.emb = nn.Linear(hidden_size * 2, vocab_size)

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device)
        c0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size)).to(device)

        return (h0, c0)

    def forward(self, x, x_lengths=None):
        # (max_seq_length // 4, batch_size, 1024)
        hidden = self.init_hidden(x.shape[1])
        if x_lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, enforce_sorted=False)

        x = self.lstm(x, hidden)[0]

        if x_lengths is not None:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        x = self.emb(x)

        return x


class SLR(nn.Module):

    def __init__(self, rnn_hidden, vocab_size, temp_fusion_type=0):
        # temp_fusion_type = >
        # 0 => input is video represented by stacking pretrained CNN features
        # 1 => input is video with fixed length represented by pretrained CNN features
        # (not 1 or 2) => input is video represented by raw frames
        super(SLR, self).__init__()
        self.vgg_s = None
        if temp_fusion_type == 0:
            self.temp_fusion = TempFusion()
        else:
            self.temp_fusion = TempFusion_FE()

        self.seq_model = BiLSTM(rnn_hidden, vocab_size)

    def forward(self, x, x_lengths=None):
        # (batch_size, max_seq_length // 4, 1024)
        x = self.temp_fusion(x)
        x = x.permute(1, 0, 2)
        # (max_seq_length // 4, batch_size, 1024)
        x = self.seq_model(x, x_lengths)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
