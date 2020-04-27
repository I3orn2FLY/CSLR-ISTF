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


class VGG_S_3D(nn.Module):
    def __init__(self):
        super(VGG_S_3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 96, (1, 7, 7), (1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
        self.lrn1 = nn.LocalResponseNorm(1)

        self.conv2 = nn.Conv3d(96, 256, (1, 5, 5), (1, 1, 1), (0, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 512, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4 = nn.Conv3d(512, 512, (1, 3, 3), (1, 1, 1), (0, 1, 1))

        self.conv5 = nn.Conv3d(512, 512, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

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
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool5(x)

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
    def __init__(self, ):
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


class TempFusion_Hand(nn.Module):
    def __init__(self):
        super(TempFusion_Hand, self).__init__()

        self.fe = VGG_S_3D()

        self.conv1d_1 = nn.Conv3d(512, 512, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv1d_2 = nn.Conv3d(512, 512, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        # (batch_size, 3, max_seq_length, 101, 101)
        x = self.fe(x)
        x = self.conv1d_1(x)
        x = self.pool1(x)
        x = self.conv1d_2(x)
        x = self.pool2(x)

        x = x.squeeze()

        x = x.permute(0, 2, 1)

        return x


class GR(nn.Module):
    def __init__(self, vocab_size, temp_fusion_type=2):
        super(GR, self).__init__()
        if temp_fusion_type == 0:
            self.temp_fusion = TempFusion()
        elif temp_fusion_type == 1:
            self.temp_fusion = TempFusion_Hand()
        elif temp_fusion_type == 2:
            self.temp_fusion = TempFusion3D()

        self.fc = nn.Linear(FRAME_FEAT_SIZE, vocab_size)

    def forward(self, x):
        x = self.temp_fusion(x)
        x = x.squeeze(1)
        x = self.fc(x)
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
        elif temp_fusion_type == 1:
            self.temp_fusion = TempFusion_Hand()
        elif temp_fusion_type == 2:
            self.temp_fusion = TempFusion3D()
        else:
            self.temp_fusion = Identity()

        self.seq_model = BiLSTM(rnn_hidden, vocab_size)

    def forward(self, x, x_lengths=None):
        # (batch_size, max_seq_length // 4, 1024)
        x = self.temp_fusion(x)
        x = x.permute(1, 0, 2)
        # (max_seq_length // 4, batch_size, 1024)
        x = self.seq_model(x, x_lengths)
        return x


class TempFusion3D(nn.Module):
    def __init__(self):
        super(TempFusion3D, self).__init__()
        self.cnn_3d = models.video.r2plus1d_18(pretrained=True)
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 7, 7))

    def forward(self, x):
        x = self.cnn_3d.stem(x)
        x = self.cnn_3d.layer1(x)
        x = self.cnn_3d.layer2(x)
        x = self.cnn_3d.layer3(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(-1, x.size(1), 1024)


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    vgg_s = GR(vocab_size=1296).to(DEVICE)
    vgg_s.eval()
    batch_size = 8
    T = 4
    C = 3
    D = 112
    with torch.no_grad():
        inp = torch.rand(batch_size, C, T, D, D).to(DEVICE)

        out = vgg_s(inp)

        print(out.shape)
