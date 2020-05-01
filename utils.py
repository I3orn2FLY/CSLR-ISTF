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

        print("\r" + something + " Progress: %.2f" % (cur_idx * 100 / self.L) + "% "
              + str(hours) + " hours "
              + str(minutes) + " minutes "
              + str(seconds) + " seconds left", end=" ")

    def end(self):
        print("\rProgress: 100%                                                                                       ")


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


class Vocab(object):
    def __init__(self):
        self.idx2gloss = ["-"]
        self.gloss2idx = {"-": 0}
        self.size = 1
        if SOURCE == "PH":
            self._build_from_PH()
        else:
            self._build_from_KSRL()

    def _build_from_PH(self):
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

    def _build_from_KSRL(self):
        with open(os.sep.join([ANNO_DIR, "vocabulary.txt"]), 'r') as f:
            lines = f.readlines()

        glosses = []
        for line in lines:
            gloss = line.strip()
            if len(gloss) < 1:
                continue
            glosses.append(gloss)

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
