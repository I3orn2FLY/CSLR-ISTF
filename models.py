import torch
import torch.nn as nn
import torchvision.models as models
from config import *
from utils import check_stf_features


class ImgFeat(nn.Module):
    def __init__(self):
        super(ImgFeat, self).__init__()
        if STF_MODEL.startswith("densenet121"):
            self.feat_m = models.densenet121(pretrained=True)
            self.feat_m.classifier = nn.Identity()
        elif STF_MODEL.startswith("googlenet"):
            self.feat_m = models.googlenet(pretrained=True)
            self.feat_m.fc = nn.Identity()
        elif STF_MODEL.startswith("resnet18"):
            self.feat_m = models.resnet18(pretrained=True)
            self.feat_m.fc = nn.Identity()
        elif STF_MODEL.startswith("pose"):
            self.feat_m = nn.Identity()
        else:
            print("Incorrect FFM", STF_MODEL)
            exit(0)

    def forward(self, x):
        return self.feat_m(x)


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

    def __init__(self, rnn_hidden, vocab_size, use_img_feat=True, use_st_feat=USE_ST_FEAT, stf_type=0):
        # temp_fusion_type = >
        # 0 => 2D temporal fusion
        # 1 => 3D temporal fusion 
        super(SLR, self).__init__()

        if use_st_feat:
            self.stf = nn.Identity()
        else:
            if stf_type == 0:
                self.stf = STF_2D(use_img_feat)
            elif stf_type == 1:
                self.stf = STF_2Plus1D()
            else:
                print("Incorrect STF type", stf_type)
                exit(0)

        self.stf_type = stf_type
        self.use_st_feat = use_st_feat
        self.use_img_feat = use_img_feat
        self.seq2seq = BiLSTM(rnn_hidden, vocab_size)

    def forward(self, x, x_lengths=None):
        # (batch_size, max_seq_length // 4, 1024)
        x = self.stf(x)
        x = x.permute(1, 0, 2)
        # (max_seq_length // 4, batch_size, 1024)
        x = self.seq2seq(x, x_lengths)
        return x


class GR(nn.Module):
    def __init__(self, vocab_size, use_feat=USE_ST_FEAT, stf_type=STF_TYPE):
        super(GR, self).__init__()
        if stf_type == 0:
            self.stf = STF_2D(use_feat=use_feat)
        elif stf_type == 1:
            self.stf = STF_2Plus1D()
        else:
            print("Incorrect temporal fusion type", stf_type)
            exit(0)

        self.fc = nn.Linear(IMG_FEAT_SIZE * 2, vocab_size)

    def forward(self, x):
        x = self.stf(x)
        x = x.view(-1, 2 * IMG_FEAT_SIZE)
        x = self.fc(x)
        return x


class STF_2Plus1D(nn.Module):
    def __init__(self):
        super(STF_2Plus1D, self).__init__()
        self.cnn = models.video.r2plus1d_18(pretrained=True)
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 7, 7))

    def forward(self, x):
        x = self.cnn.stem(x)
        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(-1, x.size(1), 1024)


class STF_2D(nn.Module):
    def __init__(self, use_feat=False):
        super(STF_2D, self).__init__()

        if use_feat:
            self.spatial_feat_m = nn.Identity()
        else:
            self.spatial_feat_m = ImgFeat()

        self.temporal_feat_m = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0)),
                                             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                                             nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0)),
                                             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.use_feat = use_feat

    def forward(self, x):
        if not self.use_feat:
            B, T, C, X, Y = x.shape
            x = x.view(B * T, C, X, Y)
            x = self.spatial_feat_m(x)
            V = x.size(1)
        else:
            B, T, V = x.shape
        x = x.view(B, 1, T, V)

        x = self.temporal_feat_m(x).squeeze(1)

        return x


# on 0th iter, when you dont load anything, and train whole network (maybe add load_stf extra var to config)
def get_end2end_model(vocab, load_seq, stf_type, use_st_feat):
    print("Loading Model... ")
    if use_st_feat and stf_type == 0:
        if not check_stf_features(img_feat=False):
            print("Can not use ST features, they are missing!")
            if check_stf_features(img_feat=True):
                print("Initializing with Image_Features")
                use_img_feat = True
                use_st_feat = False
            else:
                use_img_feat = False
                print("Can not use Img features, they are missing! (that's it boy)")
                exit(0)
        else:
            use_img_feat = False
    else:
        use_img_feat = False

    model = SLR(rnn_hidden=512, vocab_size=vocab.size,
                use_st_feat=use_st_feat, use_img_feat=use_img_feat,
                stf_type=stf_type).to(DEVICE)

    fully_loaded = False
    if os.path.exists(STF_MODEL_PATH) and not use_st_feat:
        if stf_type == 0:
            if use_img_feat:
                stf = STF_2D(False).to(DEVICE)
                stf.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
                model.stf.temporal_feat_m = stf.temporal_feat_m
                print("Temporal Features model Loaded")
                fully_loaded = True
            else:
                model.stf.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
                print("Spatiotemporal Features model Loaded")
                fully_loaded = True
        else:
            model.stf.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
            print("Spatiotemporal Features model Loaded")
            fully_loaded = True

    if os.path.exists(SEQ2SEQ_MODEL_PATH) and load_seq:
        model.seq2seq.load_state_dict(torch.load(SEQ2SEQ_MODEL_PATH, map_location=DEVICE))
        print("Seq2Seq model Loaded")
    else:
        fully_loaded = False

    return model, fully_loaded


def get_GR_model(vocab):
    model = GR(vocab.size).to(DEVICE)

    if os.path.exists(STF_MODEL_PATH):
        model.stf.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
        print("Spatiotemporal Feature Extractor Loaded")
    else:
        print("Temp fusion model doesnt exist")
        exit(0)

    return model


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    use_feat = False
    use_img_feat = True
    model = SLR(rnn_hidden=512, vocab_size=300, stf_type=STF_TYPE,
                use_img_feat=use_img_feat, use_st_feat=use_feat).to(DEVICE)
    model.eval()
    batch_size = 8
    T = 12
    C = 3
    D = 112
    with torch.no_grad():
        if use_feat:
            inp = torch.rand(batch_size, 1, T, IMG_FEAT_SIZE).to(DEVICE)
        else:
            if STF_TYPE == 0:
                if use_img_feat:
                    inp = torch.rand(batch_size, T, IMG_FEAT_SIZE).to(DEVICE)
                else:
                    inp = torch.rand(batch_size, T, C, D, D).to(DEVICE)
            else:
                inp = torch.rand(batch_size, C, T, D, D).to(DEVICE)

        out = model(inp)

        print(out.shape)
