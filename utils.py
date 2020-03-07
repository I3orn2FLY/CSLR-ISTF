import time
import numpy as np
from config import *
import pandas as pd
import pickle
import glob
import torch
import cv2
from numpy import random

from PIL import Image
from torchvision import transforms


class Vocab(object):
    def __init__(self, source="pheonix"):
        self.idx2gloss = ["-"]
        self.gloss2idx = {"-": 0}
        self.size = 1
        if source == "pheonix":
            self._build_from_pheonix()

    def _build_from_pheonix(self):
        with open(os.sep.join([ANNO_DIR, "automatic", "trainingClasses.txt"]), 'r') as f:
            lines = f.readlines()

        glosses = []
        for line in lines[1:]:
            gloss = line.split()[0]
            if gloss[-1] != '0':
                continue
            glosses.append(gloss[:-1])

        for idx, gloss in enumerate(glosses):
            self.idx2gloss.append(gloss)
            self.gloss2idx[gloss] = idx + 1

        self.size = len(self.idx2gloss)

        print("Vocabulary of length:", len(self.idx2gloss), "(blank included)")

    def encode(self, text):
        if isinstance(text, str):
            glosses = text.strip().split(" ")
        else:
            glosses = text

        return [self.gloss2idx.get(gloss, 0) for gloss in glosses]

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode(self, vectors, tensor=False):
        if tensor:
            vectors = vectors.detach().cpu().numpy()

        return [self.idx2gloss[idx] for idx in vectors]

    def decode_batch(self, vectors_seq):
        return [self.decode(vectors) for vectors in vectors_seq]


class ProgressPrinter():
    def __init__(self, L, step):
        self.start_time = time.time()
        self.L = L
        self.step = step
        self.omit_n = 0

    def omit(self):
        self.omit_n += 1

    def show(self, cur_idx):
        cur_idx -= self.omit_n - 1

        if cur_idx % self.step != 0:
            return

        time_left = int((time.time() - self.start_time) * 1.0 / cur_idx * (self.L - self.omit_n - cur_idx))

        hours = time_left // 3600

        minutes = time_left % 3600 // 60

        seconds = time_left % 60

        print("\rProgress: %.2f" % (cur_idx * 100 / self.L) + "% " \
              + str(hours) + " hours " \
              + str(minutes) + " minutes " \
              + str(seconds) + " seconds left",
              end=" ")

    def end(self):
        print("\rProgress: 100%")


def pad_features(feats, VIDEO_SEQ_LEN):
    padded_feats = np.zeros((VIDEO_SEQ_LEN, feats.shape[1]))
    L = feats.shape[0]
    if L > VIDEO_SEQ_LEN:
        step = L // VIDEO_SEQ_LEN
        left_over = L % VIDEO_SEQ_LEN
        start_idx = step // 2
        start_idxs = []

        for i in range(VIDEO_SEQ_LEN):
            start_idxs.append(start_idx)

            padded_feats[i] = feats[start_idx]
            if np.random.rand() < left_over / VIDEO_SEQ_LEN and L - 1 > start_idx + step * (VIDEO_SEQ_LEN - i - 1):
                start_idx += step + 1
            else:
                start_idx += step
    else:
        for i in range(VIDEO_SEQ_LEN):
            if i < L:
                padded_feats[i] = feats[i]
            else:
                padded_feats[i] = feats[L - 1]

    return padded_feats

