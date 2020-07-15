from config import *


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


def min_dist_transform(hypo, gt):
    dp = [[None for _ in range(len(gt) + 1)] for _ in range(len(hypo) + 1)]

    def ed(i, j):
        if dp[i + 1][j + 1] is not None:
            return dp[i + 1][j + 1]

        if i == -1 and j == -1:
            dp[0][0] = []
            return []

        if i == -1:
            # insert
            dp[0][j] = ed(-1, j - 1)
            dp[0][j + 1] = ["ins_0_" + str(gt[j])] + dp[i + 1][j]

            return dp[0][j + 1]

        if j == -1:
            # delete
            dp[i][0] = ed(i - 1, -1)

            dp[i + 1][0] = ["del_" + str(i)] + dp[i][0]

            return dp[i + 1][0]

        if hypo[i] == gt[j]:
            dp[i + 1][j + 1] = ed(i - 1, j - 1)
            return dp[i + 1][j + 1]
        else:
            # del
            dp[i][j + 1] = ed(i - 1, j)
            opt = dp[i][j + 1]
            des = "del_" + str(i)

            # ins
            dp[i + 1][j] = ed(i, j - 1)
            if len(dp[i + 1][j]) < len(opt):
                opt = dp[i + 1][j]
                des = "ins_" + str(i + 1) + "_" + str(gt[j])

            # rep
            dp[i][j] = ed(i - 1, j - 1)

            if len(dp[i][j]) < len(opt):
                opt = dp[i][j]
                des = "rep_" + str(i) + "_" + str(gt[j])

            dp[i + 1][j + 1] = [des] + opt

            return dp[i + 1][j + 1]

    return ed(len(hypo) - 1, len(gt) - 1)


def force_alignment(pred, gt):
    hypo = []
    hypo_idxs = []
    s = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            s = i + 1
            continue

        if i < len(pred) - 1 and pred[i] == pred[i + 1]:
            continue

        hypo.append(pred[i])
        hypo_idxs.append((s, i + 1))

        s = i + 1

    transform = min_dist_transform(hypo, gt)

    for oper in transform:
        oper = oper.split("_")
        des = oper[0]
        idx = int(oper[1])
        val = int(oper[-1])

        if des == "del":
            del hypo[idx]
            del hypo_idxs[idx]
        elif des == "ins":
            if idx > 0 and idx < len(hypo) - 1:
                s1, e1 = hypo_idxs[idx - 1]
                s2, e2 = hypo_idxs[idx]
                l1 = hypo[idx - 1]
                l2 = hypo[idx]
            elif idx == 0:
                s1, e1 = 0, 0
                if hypo_idxs:
                    s2, e2 = hypo_idxs[0]
                    l2 = hypo[0]
                else:
                    s2, e2 = len(pred), len(pred)
                    l2 = 0
                l1 = 0

            else:
                s1, e1 = hypo_idxs[-1]
                s2, e2 = len(pred), len(pred)
                l1 = hypo[-1]
                l2 = 0

            if l1 == val == l2:
                continue

            if s2 - e1 > 1:
                s = e1
                e = s2
                if l1 == val:
                    s += 1
                elif l2 == val:
                    e -= 1

                if e - s > 2:
                    s = s + (e - s - 2) // 2
                    e = s + 2

                hypo_idxs.insert(idx, (s, e))
                hypo.insert(idx, val)

        else:
            hypo[idx] = val

    pred = [0] * len(pred)

    for label, (s, e) in zip(hypo, hypo_idxs):
        for i in range(s, e):
            pred[i] = label

    return pred


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
