import os
import sys

sys.path.append(".." + os.sep)

from common import *
from utils import *
from models import ImgFeat, TempFusion3D
from config import *


def generate_cnn_features():
    if IMG_FEAT_MODEL.startswith("densenet") or IMG_FEAT_MODEL.startswith("googlenet"):
        mode = "2D"
        model = ImgFeat().to(DEVICE)
        preprocess = preprocess_2d
    elif IMG_FEAT_MODEL.startswith("resnet{2+1}d"):
        mode = "3D"
        model = TempFusion3D().to(DEVICE)
        if os.path.exists(GR_TF_MODEL_PATH):
            model.load_state_dict(torch.load(GR_TF_MODEL_PATH, map_location=DEVICE))

        preprocess = preprocess_3d
    else:
        print("Incorrect feature extraction model:", IMG_FEAT_MODEL)
        exit(0)

    model.eval()

    with torch.no_grad():
        generate_cnn_features_split(model, preprocess, "train", mode)
        generate_cnn_features_split(model, preprocess, "test", mode)
        generate_cnn_features_split(model, preprocess, "dev", mode)


def generate_cnn_features_split(model, preprocess, split, mode):
    if SOURCE == "KRSL" and split == "dev":
        split = "val"

    df = get_split_df(split)

    print(SOURCE, IMG_FEAT_MODEL, "feature extraction:", split, "split")
    L = df.shape[0]

    pp = ProgressPrinter(L, 10)
    for idx in range(L):
        row = df.iloc[idx]
        if SOURCE == "PH":
            video_dir = os.sep.join([VIDEOS_DIR, split, row.folder])
            feat_path = os.sep.join([VIDEO_FEAT_DIR, split, row.folder.replace("/1/*.png", ".pt")])
        else:
            video_dir = os.path.join(VIDEOS_DIR, row.video)
            feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")

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
