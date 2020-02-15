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
        if FRAME_FEAT_MODEL == "densenet121":
            self.ffm = models.densenet121(pretrained=True)
            self.ffm.classifier = Identity()
        elif FRAME_FEAT_MODEL == "googlenet":
            self.ffm = models.googlenet(pretrained=True)
            self.ffm.fc = Identity()

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
        x = self.conv1d_1(x)
        x = self.pool1(x)
        x = self.conv1d_2(x)
        x = self.pool2(x)
        return x.squeeze(1)


class TempFusion_VGG_S(nn.Module):
    def __init__(self):
        super(TempFusion_VGG_S, self).__init__()

        self.vgg_s = VGG_S()
        self.simple_temp_fusion = TempFusion()

    def forward(self, x):
        batch_feats = []
        for video_idx in range(x.shape[0]):
            video_feat = self.vgg_s(x[video_idx])
            batch_feats.append(video_feat)
        batch_feats = torch.stack(batch_feats)

        x = batch_feats.unsqueeze(1)
        x = self.simple_temp_fusion(x)
        return x


class TempFusionFixedVL(nn.Module):
    def __init__(self):
        super(TempFusionFixedVL, self).__init__()

        self.conv1d_1 = nn.Conv2d(1, 1, kernel_size=(5, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv1d_2 = nn.Conv2d(1, 1, kernel_size=(5, 1))
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
        self.lstm = nn.LSTM(input_size=FRAME_FEAT_SIZE, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.emb = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.lstm(x)[0]
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
        elif temp_fusion_type == 1:
            self.temp_fusion = TempFusionFixedVL()
        else:
            self.temp_fusion = TempFusion_VGG_S()

        self.seq_model = BiLSTM(rnn_hidden, vocab_size)

    def forward(self, x):
        x = self.temp_fusion(x)
        x = x.permute(1, 0, 2)
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
