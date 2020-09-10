import time
from config import *
import pandas as pd
import os


class ProgressPrinter():
    def __init__(self, L, step):
        self.start_time = time.time()
        self.L = L
        self.step = step
        self.omit_n = 0

    def omit(self):
        self.omit_n += 1

    def show(self, cur_idx, something=""):
        cur_idx += 1

        if cur_idx % self.step != 0:
            return

        time_left = (time.time() - self.start_time) * (self.L - cur_idx) / (cur_idx - self.omit_n)
        time_left = int(time_left)

        hours = time_left // 3600
        minutes = time_left % 3600 // 60
        seconds = time_left % 60

        if something:
            something = something + " "

        print("\r" + something + "Progress: %.2f" % (cur_idx * 100 / self.L) + "% "
              + str(hours) + " hours "
              + str(minutes) + " minutes "
              + str(seconds) + " seconds left", flush=True, end=" ")

    def end(self, something=""):
        print("\r" + something + "Progress: 100%", flush=True)



def get_split_df(split):
    if SOURCE == "PH":
        if split == "val":
            split = "dev"
        path = os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"])
        df = pd.read_csv(path, sep='|')
    else:
        if split == "dev":
            split = "val"
        path = os.sep.join([ANNO_DIR, split + ".csv"])
        df = pd.read_csv(path)
    return df


def get_video_path(row, split, stf_feat=True, feat_ext=".pt"):
    feat_dir = STF_FEAT_DIR if stf_feat else IMG_FEAT_DIR
    if SOURCE == "PH":
        video_path = os.sep.join([VIDEOS_DIR, split, row.folder.replace("/1/*.png", ".mp4")])
        feat_path = os.sep.join([feat_dir, split, row.folder.replace("/1/*.png", feat_ext)])
    elif SOURCE == "KRSL":
        video_path = os.path.join(VIDEOS_DIR, row.video)
        feat_path = os.path.join(feat_dir, row.video).replace(".mp4", feat_ext)
    else:
        print("Wrong source dataset:", SOURCE)
        exit(0)
        return None, None
    return video_path, feat_path


def check_stf_features(img_feat=False):
    print(SOURCE, STF_MODEL, "checking features...")
    for split in ["train", "dev", "test"]:

        df = get_split_df(split)

        L = df.shape[0]

        count = 0
        for idx in range(L):
            row = df.iloc[idx]
            video_path, feat_path = get_video_path(row, split, stf_feat=not img_feat)
            if os.path.exists(feat_path):
                continue
            else:
                count += 1

        if count / L > 0.05:
            return False

    return os.path.exists(STF_MODEL_PATH) or img_feat
