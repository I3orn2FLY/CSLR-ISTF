import time
from config import *
import pandas as pd
import glob
from numpy import random


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


class ProgressPrinter():
    def __init__(self, L, step):
        self.start_time = time.time()
        self.L = L
        self.step = step
        self.omit_n = 0

    def omit(self):
        self.omit_n += 1

    def show(self, cur_idx):
        cur_idx += 1

        if cur_idx % self.step != 0:
            return

        time_left = (time.time() - self.start_time) * (self.L - cur_idx) / (cur_idx - self.omit_n)
        time_left = int(time_left)

        hours = time_left // 3600
        minutes = time_left % 3600 // 60
        seconds = time_left % 60

        print("\rProgress: %.2f" % (cur_idx * 100 / self.L) + "% "
              + str(hours) + " hours "
              + str(minutes) + " minutes "
              + str(seconds) + " seconds left", end=" ")

    def end(self):
        print("\rProgress: 100%                                                                                       ")


def get_split_df(split):
    if SOURCE == "PH":
        path = os.sep.join([ANNO_DIR, "manual", split + ".corpus.csv"])
        df = pd.read_csv(path, sep='|')
    else:
        path = os.sep.join([ANNO_DIR, split + ".csv"])
        df = pd.read_csv(path)
    return df


def predict_glosses(preds, decoder):
    out_sentences = []
    if decoder:
        # need to check decoder for permutations of predictions
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(preds)
        for i in range(preds.size(0)):
            hypo = list(beam_result[i][0][:out_seq_len[i][0]])
            out_sentences.append(hypo)

    else:
        preds = preds.permute(1, 0, 2).argmax(dim=2).cpu().numpy()
        # glosses_batch = vocab.decode_batch(preds)
        for pred in preds:
            hypo = []
            for i in range(len(pred)):
                if pred[i] == 0 or (i > 0 and pred[i] == pred[i - 1]):
                    continue
                hypo.append(pred[i])

            out_sentences.append(hypo)

    return out_sentences


def gen_KRSL_annotation():
    def get_PSR(video_file):
        video_file = os.path.split(video_file)[1]
        P, S, R = video_file.split("_")[:3]
        P = int(P[1:])
        S = int(S[1:])
        R = int(R.split('.')[0])

        return P, S, R

    def get_anno_and_avoid_list():
        df = pd.read_csv(os.path.join(ANNO_DIR, "annotation.csv"), header=None)
        avoid_list = []
        vocab = set()
        with open(os.path.join(ANNO_DIR, "176_phrases"), 'r') as f:
            anno = {}
            for line in f.readlines():
                sent = line.strip().replace(',', '').lower().split()

                idx = int(sent[0])
                gloss = " ".join(df.iloc[idx][1].strip().replace(',', '').replace("(вопрос)", '').lower().split())

                trans = " ".join(df.iloc[idx][0].strip().lower().split())

                if '(' in gloss:
                    avoid_list.append(idx)
                else:
                    for word in gloss.split():
                        vocab.add(word)
                anno[int(sent[0])] = (gloss, trans)

        with open(os.path.join(ANNO_DIR, "vocabulary.txt"), 'w') as f:
            for word in sorted(list(vocab)):
                f.write(word + os.linesep)

        return anno, avoid_list

    def gen_anno_split(split_data, split_name, anno):
        P_ids = []
        S_ids = []
        videos = split_data
        glosses = []
        trans = []

        for video_path in videos:
            P, S, R = get_PSR(video_path)
            P_ids.append(P)
            S_ids.append(S)
            glosses.append(anno[S][0])
            trans.append(anno[S][1])

        df = pd.DataFrame({"P_id": P_ids, "S_id": S_ids, "video": videos, "annotation": glosses, "translation": trans})
        df.to_csv(os.path.join(ANNO_DIR, split_name + ".csv"), index=None)
        print(split_name, df.shape[0])

    anno, avoid_list = get_anno_and_avoid_list()

    video_files = os.sep.join([VIDEOS_DIR, "**", "*.*"])

    video_files = list(glob.glob(video_files))
    video_files.sort()
    data_table = {}

    video_n = 0
    for idx, video_file in enumerate(video_files):
        video_path = video_file.replace(VIDEOS_DIR + os.sep, "")

        P, S, R = get_PSR(video_file)

        if S in avoid_list:
            continue

        P_data = data_table.get(P, {})
        data_table[P] = P_data

        S_data = P_data.get(S, [])
        P_data[S] = S_data
        S_data.append(video_path)
        video_n += 1

    S_count = []
    person_count = []
    for P in data_table:
        for S in data_table[P]:
            S_count += [S] * len(data_table[P][S])
            person_count += [P] * len(data_table[P][S])

    # sents = set(list(range(176)))
    # for p in data_table:
    #     print(p, sents - set(list(data_table[p])))
    #
    # exit(0)

    P_val = int(random.rand() * 20)
    P_test = int(random.rand() * 20)
    while P_val == P_test:
        P_val = int(random.rand() * 20)
        P_test = int(random.rand() * 20)

    P_val = {P_val}
    P_test = {P_test}

    assert not P_val.intersection(P_test)

    data_train = []
    data_val = []
    data_test = []

    for P in P_val:
        for S in data_table[P]:
            data_val += data_table[P][S]

    for P in P_test:
        for S in data_table[P]:
            data_test += data_table[P][S]

    P_train = set(data_table) - P_val.union(P_test)

    S_n = (175 - len(avoid_list)) * 5 / len(P_train)

    for P in P_train:
        sents = list(data_table[P])
        random.shuffle(sents)

        for idx, S in enumerate(sents):
            R_data = data_table[P][S]
            if idx < 2 * S_n:
                rep_idx = int(random.rand() * len(R_data))

                if idx < S_n:
                    data_val.append(R_data[rep_idx])
                else:
                    data_test.append(R_data[rep_idx])

                del R_data[rep_idx]

            data_train += R_data

    random.shuffle(data_train)

    gen_anno_split(data_train, "train", anno)
    gen_anno_split(data_val, "val", anno)
    gen_anno_split(data_test, "test", anno)


if __name__ == "__main__":
    random.seed(0)
    gen_KRSL_annotation()
    vocab = Vocab()

    # print(vocab.idx2gloss)
