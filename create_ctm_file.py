import glob
import torch
from config import *
from train_utils import get_model, predict_glosses
from utils import ProgressPrinter, Vocab
import numpy as np


def decode_prediction(pred, vocab):
    pred = pred.permute(1, 0, 2).squeeze(0).argmax(dim=1).cpu().numpy()

    out_sentence = []
    start_times = []
    durations = []

    duration = 1
    step = 4
    for i in range(len(pred)):
        if pred[i] == 0 or (i > 0 and pred[i] == pred[i - 1]):
            duration += step
            continue

        durations.append(duration / 25)
        duration = step
        start_times.append(i * step / 25)
        out_sentence.append(vocab.idx2gloss[pred[i]])

    durations.append(duration / 25)
    durations = durations[1:]
    return out_sentence, start_times, durations


def create_ctm_file_split(split):
    vocab = Vocab()
    model, loaded, model_path = get_model(END2END_TRAIN_MODE, vocab, END2END_TRAIN_LOAD)
    model.eval()

    gt_ctm_val_file = os.sep.join([PH_EVA_DIR, "phoenix2014-groundtruth-" + split + ".stm"])

    with open(gt_ctm_val_file, 'r') as f:
        lines = f.readlines()

    dirs = []
    for line in lines:
        dirs.append(line.split(" ")[0])

    prefix = VIDEO_FEAT_DIR
    out_ctm_file = model_path.replace(model_path.split("_")[-1], split + ".ctm")
    out_ctm_file = os.path.split(out_ctm_file)[1]

    with open(os.sep.join([PH_EVA_DIR, out_ctm_file]), 'w') as f:

        with torch.no_grad():
            for idx, dir in enumerate(dirs):
                feat_path = os.sep.join([prefix, split, dir, "1.npy"])
                feats = np.load(feat_path)
                inp = torch.Tensor(feats).to(DEVICE).unsqueeze(0)
                if END2END_TRAIN_MODE == "FULL":
                    inp = inp.unsqueeze(1)

                pred = model(inp).log_softmax(dim=2)

                out_sentence, start_times, durations = decode_prediction(pred, vocab)
                for gloss, start_time, duration in zip(out_sentence, start_times, durations):
                    f.write(" ".join([dir, "1", "%.3f" % start_time, "%.3f" % duration, gloss]) + os.linesep)


if __name__ == "__main__":
    create_ctm_file_split("test")
