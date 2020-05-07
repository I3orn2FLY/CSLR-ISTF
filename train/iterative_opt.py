from train.train_end2end import *
from train.train_gloss_recog import *

# TODO implement iterative opt


if __name__ == "__main__":
    vocab = Vocab()
    end2end_model = get_end2end_model(vocab)
    datasets = get_end2end_model(vocab)
    train(model, vocab, datasets)
