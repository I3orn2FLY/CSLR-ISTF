from models import get_end2end_model, get_GR_model
from dataset import get_gr_datasets, get_end2end_datasets
from train.end2end import train_end2end
from train.gloss_recog import train_gloss_recog
from utils import Vocab
from config import *

# TODO implement iterative opt


if __name__ == "__main__":
    vocab = Vocab()
    datasets = get_end2end_datasets(vocab)
    model, _ = get_end2end_model(vocab, load=END2END_MODEL_LOAD, stf_type=STF_TYPE, use_feat=False)
    train_end2end(model, vocab, datasets, USE_STF_FEAT)

    model = get_GR_model(vocab)
    datasets = get_gr_datasets()
    train_gloss_recog(model, datasets)
