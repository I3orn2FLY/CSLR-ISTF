import glob
import os
import cv2
from config import PH_DIR, VIDEOS_DIR
from utils import ProgressPrinter


# Converting into folders with images into video files
# Because it is much faster to read from cv2.VideoCapture rather than using imread on each image in folder

def convert_phoenix_to_videos():
    if SRC_MODE == "FULL":
        ph_images_dir = os.sep.join([PH_DIR, "features", "fullFrame-210x260px"])
    else:
        ph_images_dir = None
        exit(0)

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


if __name__ == "__main__":
    convert_phoenix_to_videos()
