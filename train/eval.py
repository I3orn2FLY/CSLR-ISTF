import torch
import sys

sys.path.append("..")
from config import *
from models import get_end2end_model
from vocab import Vocab
from utils import ProgressPrinter, get_split_df, get_video_path
import Levenshtein as Lev


def eval_split_by_lev(model, vocab, split):
    df = get_split_df(split)
    pp = ProgressPrinter(df.shape[0], 5)
    hypes = []
    gts = []
    with torch.no_grad():
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            gt = vocab.encode(row.annotation)
            video_path, feat_path = get_video_path(row, split)
            tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
            pred = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1).cpu().numpy()

            hypo = []
            for i in range(len(pred)):
                if pred[i] == 0 or (i > 0 and pred[i] == pred[i - 1]):
                    continue
                hypo.append(pred[i])

            gts += gt
            hypes += hypo
            pp.show(idx)

        pp.end()

        hypes = "".join([chr(x) for x in hypes])
        gts = "".join([chr(x) for x in gts])
        wer = Lev.distance(hypes, gts) / len(gts) * 100

        print(wer)


def decode_prediction(pred, vocab):
    out_sentence = []
    start_times = []
    durations = []

    step = 4
    for i in range(len(pred)):
        if pred[i] == 0:
            continue

        if (i > 0 and pred[i] == pred[i - 1]):
            durations[-1] += step / 25
        else:

            durations.append(step / 25)
            start_times.append(i * step / 25)
            out_sentence.append(vocab.idx2gloss[pred[i]])

    return out_sentence, start_times, durations


def create_ctm_file_split(model, vocab, split):
    gt_ctm_val_file = os.sep.join([PH_EVA_DIR, "phoenix2014-groundtruth-" + split + ".stm"])

    with open(gt_ctm_val_file, 'r') as f:
        lines = f.readlines()

    dirs = []
    for line in lines:
        dirs.append(line.split(" ")[0])

    out_ctm_path = STF_MODEL + "_" + split + ".ctm"
    with open(os.sep.join([PH_EVA_DIR, out_ctm_path]), 'w') as f:
        with torch.no_grad():
            for idx, dir in enumerate(dirs):
                feat_path = os.sep.join([STF_FEAT_DIR, split, dir + ".pt"])
                inp = torch.load(feat_path).to(DEVICE).unsqueeze(0)

                pred = model(inp)

                pred = pred.squeeze(1).log_softmax(dim=1).argmax(dim=1).cpu().numpy()

                out_sentence, start_times, durations = decode_prediction(pred, vocab)

                for gloss, start_time, duration in zip(out_sentence, start_times, durations):
                    f.write(" ".join([dir, "1", "%.3f" % start_time, "%.3f" % duration, gloss]) + os.linesep)


if __name__ == "__main__":
    vocab = Vocab()
    model, loaded = get_end2end_model(vocab, True, 1, True)
    model.eval()
    with torch.no_grad():
        create_ctm_file_split(model, vocab, "dev")
        eval_split_by_lev(model, vocab, "dev")
        create_ctm_file_split(model, vocab, "test")
        eval_split_by_lev(model, vocab, "test")
