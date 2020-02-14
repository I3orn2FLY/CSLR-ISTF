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
        rand_idx = int(np.random.rand() * L)
        for idx in range(L):
            row = df.iloc[idx]
            img_dir = os.sep.join([IMAGES_DIR, split, row.folder])
            feat_dir = os.sep.join([VIDEO_FEAT_DIR, split, row.folder])
            feat_file = feat_dir.replace("/*.png", "")

            if os.path.exists(feat_file + ".npy"):
                continue

            feat_dir = os.path.split(feat_file)[0]

            image_files = list(glob.glob(img_dir))
            image_files.sort()

            # image_files = [os.path.split(img_file)[1] for img_file in image_files]
            # if idx == rand_idx:
            #     print()
            #     for img_file in image_files:
            #         print(img_file.split('.')[1].split('_')[-1])
            #
            #     exit()

            images = [Image.open(img_file) for img_file in image_files]
            inp = torch.stack([preprocess(image) for image in images])
            inp = inp.to(device)
            feats = model(inp).cpu().numpy()

            if not os.path.exists(feat_dir):
                os.makedirs(feat_dir)
            np.save(feat_file, feats)

            if idx % 10 == 0:
                print_progress(idx + 1, L, start_time)

        print()


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = FrameFeatModel().to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    generate_split(model, device, preprocess, "train")
    generate_split(model, device, preprocess, "test")
    generate_split(model, device, preprocess, "dev")
