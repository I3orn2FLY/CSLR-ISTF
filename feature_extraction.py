import torch
from torchvision import transforms
import glob
from PIL import Image
import time
import numpy as np
import pandas as pd
from models import FrameFeatModel
from config import *
from utils import print_progress


def generate_split(model, device, preprocess, split):
    df = pd.read_csv(os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"]), sep='|')
    print("Feature extraction:", split, "split")
    with torch.no_grad():
        L = df.shape[0]
        start_time = time.time()
        for idx in range(L):
            row = df.iloc[idx]
            img_dir = os.sep.join([IMAGES_DIR, split, row.folder])
            feat_dir = os.sep.join([VIDEO_FEAT_DIR, split, row.folder])
            feat_file = feat_dir.replace("/*.png", "")

            if os.path.exists(feat_file + ".npy"):
                continue

            feat_dir = os.path.split(feat_file)[0]
            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)

            image_files = list(glob.glob(img_dir))
            images = [Image.open(img_file) for img_file in image_files]
            inp = torch.stack([preprocess(image) for image in images])
            inp = inp.to(device)

            feats = model(inp).cpu().numpy()
            np.save(feat_file, feats)

            if idx % 10 == 0:
                print_progress(idx + 1, L, start_time)

        print()


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = FrameFeatModel()
    if FRAME_FEAT_MODEL == "densenet121":
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = None

    generate_split(model, device, preprocess, "train")
    generate_split(model, device, preprocess, "test")
    generate_split(model, device, preprocess, "dev")
