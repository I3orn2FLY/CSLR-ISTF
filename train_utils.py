import torch
from torch.optim import Adam, RMSprop, SGD
from models import SLR, weights_init
from dataset import PhoenixEnd2EndDataset

from config import *


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


def get_wer_info(mode, loaded):
    best_wer_dir = os.sep.join([VARS_DIR, "PheonixWER"])
    if not os.path.exists(best_wer_dir):
        os.makedirs(best_wer_dir)

    if mode == "Full":
        criterion_phase = END2END_FULL_CRIT_PHASE
        best_wer_file = os.sep.join([best_wer_dir, "end2end_full_" + FRAME_FEAT_MODEL + "_" + criterion_phase + ".txt"])
    else:
        criterion_phase = END2END_HAND_CRIT_PHASE
        best_wer_file = os.sep.join([best_wer_dir, "end2end_hand_" + criterion_phase + ".txt"])

    if os.path.exists(best_wer_file) and loaded:
        with open(best_wer_file, 'r') as f:
            best_wer = float(f.readline().strip())
            print("BEST " + criterion_phase + " WER:", best_wer)
    else:
        best_wer = float("inf")
        print("BEST " + criterion_phase + " WER:", "Not found")

    return best_wer, best_wer_file, criterion_phase


def get_train_info(mode, model):
    if mode == "Full":
        n_epochs = END2END_FULL_N_EPOCHS
        lr = END2END_FULL_LR
    else:
        n_epochs = END2END_HAND_N_EPOCHS
        lr = END2END_HAND_LR



    if END2END_TRAIN_OPTIMIZER == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif END2END_TRAIN_OPTIMIZER == "RMSProp":
        optimizer = RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr)

    return optimizer, n_epochs


def save_model(model, model_path, best_wer, best_wer_file):
    with open(best_wer_file, 'w') as f:
        f.write(str(best_wer) + "\n")

    torch.save(model.state_dict(), model_path)
    print("Model Saved")


def get_model(mode, vocab, load=False):
    if mode == "Full":
        temp_fusion_type = 0
        model_path = END2END_FULL_MODEL_PATH
    else:
        temp_fusion_type = 2
        model_path = END2END_HAND_MODEL_PATH

    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=temp_fusion_type).to(DEVICE)
    if load and os.path.exists(model_path):
        loaded = True
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model Loaded")
    else:
        loaded = False
        model.apply(weights_init)

    return model, loaded, model_path


def get_datasets(mode, vocab):
    if mode == "Full":
        batch_size = END2END_FULL_BATCH_SIZE
        aug_temp = END2END_FULL_AUG_TEMP
        aug_frame = False
    else:
        batch_size = END2END_HAND_BATCH_SIZE
        aug_temp = END2END_HAND_AUG_TEMP
        aug_frame = END2END_HAND_AUG_FRAME

    tr_dataset = PhoenixEnd2EndDataset(mode, vocab, "train", max_batch_size=batch_size,
                                       augment_temp=aug_temp, augment_frame=aug_frame)

    val_dataset = PhoenixEnd2EndDataset(mode, vocab, "dev", max_batch_size=batch_size)

    datasets = {"Train": tr_dataset, "Val": val_dataset}

    return datasets
