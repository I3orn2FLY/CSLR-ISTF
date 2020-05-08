from common import *
from utils import *
from models import STF_2D, STF_2Plus1D, SLR
from config import *


def generate_cnn_features():
    if not os.path.exists(STF_MODEL_PATH):
        print("STF model doesnt exist:", STF_MODEL_PATH)
        exit(0)

    if STF_MODEL.startswith("densenet") or STF_MODEL.startswith("googlenet"):
        mode = "2D"
        model = STF_2D(use_feat=False).to(DEVICE)
        preprocess = preprocess_2d
    elif STF_MODEL.startswith("resnet{2+1}d"):
        mode = "3D"
        model = STF_2Plus1D().to(DEVICE)
        preprocess = preprocess_3d
    else:
        model = None
        preprocess = None
        mode = None
        print("Incorrect feature extraction model:", STF_MODEL)
        exit(0)

    model.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        generate_cnn_features_split(model, preprocess, "train", mode)
        generate_cnn_features_split(model, preprocess, "test", mode)
        generate_cnn_features_split(model, preprocess, "dev", mode)


def generate_cnn_features_split(model, preprocess, split, mode):
    if SOURCE == "KRSL" and split == "dev":
        split = "val"

    df = get_split_df(split)

    print(SOURCE, STF_MODEL, "feature extraction:", split, "split")
    L = df.shape[0]

    pp = ProgressPrinter(L, 10)
    for idx in range(L):
        row = df.iloc[idx]
        if SOURCE == "PH":
            video_dir = os.sep.join([VIDEOS_DIR, split, row.folder])
            feat_path = os.sep.join([STF_FEAT_DIR, split, row.folder.replace("/1/*.png", ".pt")])
        else:
            video_dir = os.path.join(VIDEOS_DIR, row.video)
            feat_path = os.path.join(STF_FEAT_DIR, row.video).replace(".mp4", ".pt")

        if os.path.exists(feat_path) and not FEAT_OVERRIDE:
            pp.omit()
            continue

        feat_dir = os.path.split(feat_path)[0]

        images = get_images(video_dir)
        if len(images) < 4:
            continue

        tensor_video = get_tensor_video(images, preprocess, mode)
        if mode == "2D":
            inp = tensor_video.to(DEVICE)
            feat = model(inp).cpu()
        else:
            inp = tensor_video.unsqueeze(0).to(DEVICE)
            feat = model(inp).squeeze(0).cpu()

        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        torch.save(feat, feat_path)

        if SHOW_PROGRESS:
            pp.show(idx)

    if SHOW_PROGRESS:
        pp.end()


if __name__ == "__main__":
    generate_cnn_features()
