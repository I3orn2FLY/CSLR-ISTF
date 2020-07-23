import torch

from processing_tools import preprocess_2d, preprocess_3d, get_images, get_tensor_video
from utils import ProgressPrinter, get_video_path, get_split_df
from models import STF_2D, STF_2Plus1D
from config import *


def generate_stf_feats(stf_model=STF_MODEL):
    if not os.path.exists(STF_MODEL_PATH):
        print("STF model doesnt exist:", STF_MODEL_PATH)
        exit(0)

    if stf_model.startswith("densenet") or stf_model.startswith("googlenet"):
        mode = "2D"
        model = STF_2D().to(DEVICE)
        preprocess = preprocess_2d

    elif stf_model.startswith("resnet{2+1}d"):
        mode = "3D"
        model = STF_2Plus1D().to(DEVICE)
        preprocess = preprocess_3d
        model.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
    else:
        model = None
        preprocess = None
        mode = None
        print("Incorrect feature extraction model:", stf_model)
        exit(0)

    model.load_state_dict(torch.load(STF_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(SOURCE, stf_model, "SpatioTemporal feature extraction...")
    with torch.no_grad():

        gen_stf_feats_split(model, preprocess, "train", mode)
        gen_stf_feats_split(model, preprocess, "test", mode)
        gen_stf_feats_split(model, preprocess, "dev", mode)


def gen_stf_feats_split(model, preprocess, split, mode):
    if SOURCE == "KRSL" and split == "dev":
        split = "val"

    df = get_split_df(split)

    L = df.shape[0]
    print(split, "split")
    pp = ProgressPrinter(L, 10)
    for idx in range(L):
        row = df.iloc[idx]
        video_path, feat_path = get_video_path(row, split)

        if os.path.exists(feat_path) and not FEAT_OVERRIDE:
            pp.omit()
            continue

        feat_dir = os.path.split(feat_path)[0]

        images = get_images(video_path)
        if len(images) < 4:
            continue

        tensor_video = get_tensor_video(images, preprocess, mode)
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
    generate_stf_feats()
