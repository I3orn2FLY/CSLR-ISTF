from common import *
from utils import *
from models import ImgFeat
from config import *


def generate_img_feats():
    model = None
    preprocess = None
    if STF_MODEL.startswith("densenet") or STF_MODEL.startswith("googlenet"):
        model = ImgFeat().to(DEVICE)
        preprocess = preprocess_2d
    else:
        print("Incorrect feature extraction model:", STF_MODEL)
        exit(0)


    model.eval()

    with torch.no_grad():
        gen_img_feat_split(model, preprocess, "train")
        gen_img_feat_split(model, preprocess, "test")
        gen_img_feat_split(model, preprocess, "dev")


def gen_img_feat_split(model, preprocess, split, mode):
    if SOURCE == "KRSL" and split == "dev":
        split = "val"

    df = get_split_df(split)

    print(SOURCE, STF_MODEL, "feature extraction:", split, "split")
    L = df.shape[0]

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
    generate_img_feats()
