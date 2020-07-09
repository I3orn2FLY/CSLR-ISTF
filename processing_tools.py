from config import *
import torch
import numpy as np
import cv2


def preprocess_img(img, mean, std):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255
    return (img - mean) / std


def preprocess_2d(img):
    if img.shape[:2] != (IMG_SIZE_2D, IMG_SIZE_2D):
        img = cv2.resize(img, (IMG_SIZE_2D, IMG_SIZE_2D))

    img = preprocess_img(img, np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))

    return img


def preprocess_3d(img):
    if img.shape[:2] != (IMG_SIZE_2Plus1D, IMG_SIZE_2Plus1D):
        img = cv2.resize(img, (IMG_SIZE_2Plus1D, IMG_SIZE_2Plus1D))

    img = preprocess_img(img, np.array([0.43216, 0.394666, 0.37645]), np.array([0.22803, 0.22145, 0.216989]))

    return img


def get_images(video_path, size=None):
    images = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        if size is not None:
            img = cv2.resize(img, size)
        images.append(img)
    cap.release()

    return images


def get_tensor_video(images, preprocess, mode):
    video = []
    for img in images:
        img = preprocess(img)
        video.append(img)

    video_tensor = np.stack(video).astype(np.float32)
    if mode == "2D":
        axes = [0, 3, 1, 2]
    else:
        axes = [3, 0, 1, 2]
    video_tensor = video_tensor.transpose(axes)
    video_tensor = torch.from_numpy(video_tensor)

    return video_tensor








if __name__ == "__main__":
    np.random.seed(0)

    # print(vocab.idx2gloss)
