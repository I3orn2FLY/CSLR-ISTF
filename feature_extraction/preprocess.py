# TODO make image mean and std calculating and loading
import glob
import os
import cv2
from config import SRC_MODE, PH_DIR, VIDEOS_DIR
from utils import ProgressPrinter


def convert_phoenix_to_videos():
    if SRC_MODE == "FULL":
        ph_images_dir = os.sep.join([PH_DIR, "features", "fullFrame-210x260px"])
    elif SRC_MODE == "HAND":
        ph_images_dir = os.sep.join([PH_DIR, "features", "trackedRightHand-92x132px"])
    else:
        ph_images_dir = None
        exit(0)

    video_dirs = list(glob.glob(os.sep.join([ph_images_dir, '*', '*', '1'])))

    pp = ProgressPrinter(len(video_dirs), 5)
    print("Converting Images into Videos", SRC_MODE)
    for idx, video_dir in enumerate(video_dirs):
        image_paths = sorted(list(glob.glob(os.path.join(video_dir, "*.png"))))
        video_path = os.path.split(video_dir)[0] + ".mp4"
        video_path = os.sep.join([VIDEOS_DIR] + video_path.split(os.sep)[-2:])
        if os.path.exists(video_path): continue
        video_dir = os.path.split(video_path)[0]
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if SRC_MODE == "FULL":
            shape = (210, 260)
        else:
            shape = (92, 132)

        out = cv2.VideoWriter(video_path, fourcc, 25.0, shape)
        for im in image_paths:
            frame = cv2.imread(im)
            out.write(frame)

        out.release()
        pp.show(idx)

    pp.end()

    print()


if __name__ == "__main__":
    convert_phoenix_to_videos()
