import sys
import os
import glob
import numpy as np
import pandas as pd

sys.path.append("..")

from config import ANNO_DIR, VIDEOS_DIR


def gen_KRSL_annotation():
    def get_PSR(video_file):
        video_file = os.path.split(video_file)[1]
        P, S, R = video_file.split("_")[:3]
        P = int(P[1:])
        S = int(S[1:])
        R = int(R.split('.')[0])

        return P, S, R

    def get_anno_and_avoid_list():
        def preprocess_sentence(sent):
            return " ".join(sent.strip().replace(',', '').replace("(вопрос)", '').lower().split())

        df = pd.read_csv(os.path.join(ANNO_DIR, "annotation.csv"), sep="|")

        # df.to_csv(os.path.join(ANNO_DIR, "annotation.csv"), sep="|")
        avoid_list = []
        vocab = set()

        anno = []
        for idx in range(df.shape[0]):
            trans = preprocess_sentence(df.iloc[idx]["Translation"])
            gloss = preprocess_sentence(df.iloc[idx]["Recognition"])

            if '(' in gloss:
                avoid_list.append(idx)
            else:
                for word in gloss.split():
                    vocab.add(word)
            anno.append((gloss, trans))

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

    P_val = int(np.random.rand() * 20)
    P_test = int(np.random.rand() * 20)
    while P_val == P_test:
        P_val = int(np.random.rand() * 20)
        P_test = int(np.random.rand() * 20)

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
        np.random.shuffle(sents)

        for idx, S in enumerate(sents):
            R_data = data_table[P][S]
            if idx < 2 * S_n:
                rep_idx = int(np.random.rand() * len(R_data))

                if idx < S_n:
                    data_val.append(R_data[rep_idx])
                else:
                    data_test.append(R_data[rep_idx])

                del R_data[rep_idx]

            data_train += R_data

    np.random.shuffle(data_train)

    gen_anno_split(data_train, "train", anno)
    gen_anno_split(data_val, "val", anno)
    gen_anno_split(data_test, "test", anno)


if __name__ == "__main__":
    gen_KRSL_annotation()
