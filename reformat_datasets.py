import glob
import os
import cv2
import numpy as np
from config import PH_DIR, VIDEOS_DIR, KRSL_DIR, ANNO_DIR
from utils import ProgressPrinter, get_split_df, get_video_path


# Converting into folders with images into video files
# Because it is much faster to read from cv2.VideoCapture rather than using imread on each image in folder

def convert_phoenix_to_videos():
    ph_images_dir = os.sep.join([PH_DIR, "features", "fullFrame-210x260px"])

    video_dirs = list(glob.glob(os.sep.join([ph_images_dir, '*', '*', '1'])))

    pp = ProgressPrinter(len(video_dirs), 5)
    print("Converting Images into Videos")
    for idx, video_dir in enumerate(video_dirs):
        image_paths = sorted(list(glob.glob(os.path.join(video_dir, "*.png"))))
        video_path = os.path.split(video_dir)[0] + ".mp4"
        video_path = os.sep.join([VIDEOS_DIR] + video_path.split(os.sep)[-2:])
        if os.path.exists(video_path):
            pp.omit()
            continue
        video_dir = os.path.split(video_path)[0]
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        shape = (210, 260)

        out = cv2.VideoWriter(video_path, fourcc, 25.0, shape)
        for im in image_paths:
            frame = cv2.imread(im)
            out.write(frame)

        out.release()
        pp.show(idx)

    pp.end()

    print()


def get_foreground_coords(frame):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w, c = frame.shape
    for cnt in contours:
        rect = cv2.boundingRect(cnt)

        alias_w = w - rect[2] - rect[0]
        if (alias_w - 10 > rect[0] or rect[0] > alias_w + 10):
            continue

        alias_h = h - rect[3] - rect[1]

        if (alias_h - 10 > rect[1] or rect[1] > alias_h + 10):
            continue

        fr = rect[2] * rect[3] / (w * h)

        if fr < 0.3 or fr > 0.9:
            continue

        return rect[1], rect[1] + rect[3], rect[0], rect[0] + rect[2]

    return None


def resize_images(images):
    h, w = images[-1].shape[:2]
    ratio = w / h - 1

    if ratio > 0.3:
        new_w = 360
        new_h = 200
    elif ratio < - 0.2:
        new_w = 200
        new_h = 360
    else:
        new_w = 200
        new_h = 200

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (new_w, new_h))

    return images


def get_images(video):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    coords = None
    images = []
    finished = False

    coords_count = {}
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            finished = True
            break

        if not coords:
            coords = get_foreground_coords(frame)
            if coords:
                coords_count[coords] = coords_count.get(coords, 0) + 1

        if coords:
            frame = frame[coords[0]:coords[1], coords[2]: coords[3]]

        images.append(frame)

    max_count = 0
    for coord in coords_count:
        if coords_count[coord] > max_count:
            max_count = coords_count[coord]
            coords = coord

    if coords is not None:
        new_w = coords[3] - coords[2]
        new_h = coords[1] - coords[0]
        i = 0
        while (i < len(images)):
            if (images[i].shape[:2] == (h, w)):
                del images[i]
                continue
            if (images[i].shape[:2] != (new_h, new_w)):
                images[i] = cv2.resize(images[i], (new_w, new_h))
            i += 1

        w, h = new_w, new_h

    if not finished:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if coords:
                frame = frame[coords[0]:coords[1], coords[2]: coords[3]]

            images.append(frame)

    cap.release()
    return images, fps


def clean_anno_KRSL(split, save=True):
    df = get_split_df(split)
    L = df.shape[0]
    to_remove = []
    for i in range(L):
        row = df.iloc[i]

        video_path, _ = get_video_path(row, split)
        if not os.path.exists(video_path):
            to_remove.append(i)

    df = df.drop(df.index[to_remove])
    if save:
        df.to_csv(os.path.join(ANNO_DIR, split + ".csv"), index=None)

    print("Cleaned ", split, "dataset, from", L, "to", df.shape[0])


def reformat_KRSL():
    np.random.seed(0)
    krsl_video_dir = os.path.join(KRSL_DIR, "videos")

    videos = list(glob.glob(os.sep.join([krsl_video_dir, "**", "*.mp4"])))

    fps_out = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    np.random.shuffle(videos)
    pp = ProgressPrinter(len(videos), 15)

    print("Reformatting KRSL")

    not_images = 0
    for idx, video_path in enumerate(videos):
        out_video_path = os.sep.join([VIDEOS_DIR] + video_path.split(os.sep)[-2:])
        video_dir = os.path.split(out_video_path)[0]
        if (os.path.exists(out_video_path)):
            pp.omit()
            continue

        images, fps = get_images(video_path)

        if not images:
            not_images += 1
            pp.omit()
            continue

        images = resize_images(images)

        L = len(images)

        L_out = round(L * fps_out / fps)

        images = np.array(images)

        idxs = np.linspace(0, L, L_out, endpoint=False)

        hw = images.shape[1:3]
        assert (hw == (200, 360) or hw == (200, 200) or hw == (360, 200))
        images = [images[round(i)] for i in idxs]

        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        out = cv2.VideoWriter(out_video_path, fourcc, 25.0, hw[::-1])
        for frame in images:
            out.write(frame)

        out.release()
        pp.show(idx)
    pp.end()
    clean_anno_KRSL("train", save=True)
    clean_anno_KRSL("test", save=True)
    clean_anno_KRSL("dev", save=True)


if __name__ == "__main__":
    # convert_phoenix_to_videos()
    reformat_KRSL()
