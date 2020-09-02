import matplotlib.pyplot as plt
import os
import sys
import pickle
import numpy as np

import sys

sys.path.append("..")
from config import *

if __name__ == "__main__":
    with open(os.path.join(VARS_DIR, "curve.pkl"), 'rb') as f:
        curve = pickle.load(f)

    plt.plot(range(1, len(curve["Train"]) + 1), curve["Train"], label="Train")
    plt.plot(range(1, len(curve["Val"]) + 1), curve["Val"], label="Val")
    min_epoch = np.argmin(curve["Val"]) + 1

    train_wer = curve["Train"][min_epoch - 1]

    val_wer = (np.min(curve["Val"]))
    plt.plot([min_epoch, min_epoch], [0, val_wer], '--')
    plt.plot([0, min_epoch], [val_wer, val_wer], '--')
    plt.plot([0, min_epoch], [train_wer, train_wer], '--')
    plt.xlabel("N epoch")
    plt.ylabel("WER", rotation=0, x=0.1, y=0.47)
    plt.title("Learning Curve")

    plt.ylim(bottom=0)
    plt.xlim(left=-1)
    plt.yticks([train_wer, val_wer] + list(range(10, 91, 20)))
    plt.xticks([min_epoch] + list(range(20, 201, 20)))

    plt.legend()
    plt.savefig(os.path.join(VARS_DIR, "learning_curve.png"))
    plt.show()
