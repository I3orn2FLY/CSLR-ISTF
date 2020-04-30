import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import PIL
from config import *


class ImgFeat(nn.Module):
    def __init__(self):
        super(ImgFeat, self).__init__()
        if IMG_FEAT_MODEL.startswith("densenet121"):
            self.feat_m = models.densenet121(pretrained=True)
            self.feat_m.classifier = nn.Identity()
        elif IMG_FEAT_MODEL.startswith("googlenet"):
            self.feat_m = models.googlenet(pretrained=True)
            self.feat_m.fc = nn.Identity()
        elif IMG_FEAT_MODEL.startswith("resnet18"):
            self.feat_m = models.resnet18(pretrained=True)
            self.feat_m.fc = nn.Identity()
        elif IMG_FEAT_MODEL.startswith("vgg-s"):
            self.feat_m = VGG_S
        else:
            print("Incorrect FFM", IMG_FEAT_MODEL)
            exit(0)

    def forward(self, x):
        return self.feat_m(x)


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


class GR(nn.Module):
    def __init__(self, vocab_size, use_feat=INP_FEAT, temp_fusion_type=1):
        super(GR, self).__init__()
        if temp_fusion_type == 0:
            self.temp_fusion = TempFusion2D(use_feat=use_feat)
        elif temp_fusion_type == 1:
            self.temp_fusion = TempFusion3D()
        else:
            print("Incorrect temporal fusion type", temp_fusion_type)
            exit(0)

        self.fc = nn.Linear(IMG_FEAT_SIZE * 2, vocab_size)

    def forward(self, x):
        x = self.temp_fusion(x)
        x = x.view(-1, 2 * IMG_FEAT_SIZE)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=IMG_FEAT_SIZE, hidden_size=hidden_size, num_layers=num_layers,
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

    def __init__(self, rnn_hidden, vocab_size, use_feat=INP_FEAT, temp_fusion_type=0):
        # temp_fusion_type = >
        # 0 => 2D temporal fusion
        # 1 => 3D temporal fusion 
        super(SLR, self).__init__()
        self.vgg_s = None
        if temp_fusion_type == 0:
            self.temp_fusion = TempFusion2D(use_feat=use_feat)
        elif temp_fusion_type == 1:
            if use_feat:
                self.temp_fusion = nn.Identity()
            else:
                self.temp_fusion = TempFusion3D()
        else:
            print("Incorrect temporal fusion type", temp_fusion_type)
            exit(0)

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


class TempFusion2D(nn.Module):
    def __init__(self, use_feat):
        super(TempFusion2D, self).__init__()
        if use_feat:
            self.feat_m = nn.Identity()
        else:
            self.feat_m = ImgFeat()

        self.tf = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0)),
                                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                                nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0)),
                                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))

        self.use_feat = use_feat

    def forward(self, x):
        if not self.use_feat:
            B, T, C, X, Y = x.shape
            x = x.view(B * T, C, X, Y)
            x = self.feat_m(x)
            V = x.size(1)
            x = x.view(B, 1, T, V)

        x = self.tf(x).squeeze(1)

        return x.permute(1, 0, 2)


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    use_feat = False
    model = SLR(rnn_hidden=512, vocab_size=300, temp_fusion_type=1, use_feat=use_feat).to(DEVICE)
    model.eval()
    batch_size = 8
    T = 12
    C = 3
    D = 112
    with torch.no_grad():
        if use_feat:
            inp = torch.rand(batch_size, 1, T, IMG_FEAT_SIZE).to(DEVICE)
        else:
            inp = torch.rand(batch_size, C, T, D, D).to(DEVICE)

        out = model(inp)

        print(out.shape)
