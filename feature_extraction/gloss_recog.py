import sys
import os

sys.path.append(".." + os.sep)

from utils import *
from models import SLR
from common import *


def save_glosses(images, pad_image, gloss_idx, stride):
    gloss_paths = []

    s = 0
    p = stride // 2

    images = p * [pad_image] + images + p * [pad_image]
    while s < len(images):
        e = min(len(images), s + 2 * stride)
        if e - s > stride:
            gloss_images = images[s:e]
            gloss_path = os.path.join(GLOSS_DATA_DIR, str(gloss_idx))
            if not os.path.exists(gloss_path):
                os.makedirs(gloss_path)

            # for idx, image in enumerate(gloss_images):
            #     cv2.imwrite(os.path.join(gloss_path, str(idx) + ".jpg"), image)

            gloss_paths.append(os.path.join(str(gloss_idx), "*.jpg"))

            gloss_idx += 1

        s += stride

    return gloss_paths


def generate_gloss_dataset():
    vocab = Vocab()
    if not IMG_FEAT_MODEL.startswith("resnet{2+1}d") or TEMP_FUSION_TYPE != 1:
        print("Incorrect feature extraction model:", IMG_FEAT_MODEL, TEMP_FUSION_TYPE)
        exit(0)

    model = SLR(rnn_hidden=512, vocab_size=vocab.size, temp_fusion_type=1).to(DEVICE)

    if os.path.exists(END2END_MODEL_PATH):
        model.load_state_dict(torch.load(END2END_MODEL_PATH, map_location=DEVICE))
        print("Model Loaded")
    else:
        print("Model doesnt exist")
        exit(0)

    model.eval()

    pad_image = 255 * np.ones((IMG_SIZE_3D, IMG_SIZE_3D, 3)) * np.array([0.406, 0.485, 0.456])

    pad_image = pad_image.astype(np.uint8)

    temp_stride = 4
    df = get_split_df("train")
    Y = []
    gloss_paths = []
    with torch.no_grad():

        pp = ProgressPrinter(df.shape[0], 5)
        gloss_idx = 0

        for idx in range(df.shape[0]):

            row = df.iloc[idx]

            if SOURCE == "PH":
                video_dir = os.sep.join([VIDEOS_DIR, "train", row.folder])
            elif SOURCE == "KRSL":
                video_dir = os.path.join(VIDEOS_DIR, row.video)
            else:
                print("Wrong Dataset:", SOURCE)
                exit(0)
            images = get_images(video_dir, size=(IMG_SIZE_3D, IMG_SIZE_3D))
            if len(images) < 4:
                continue

            if INP_FEAT:
                if SOURCE == "PH":
                    feat_path = os.sep.join([VIDEO_FEAT_DIR, "train", row.folder.replace("/1/*.png", ".pt")])
                elif SOURCE == "KRSL":
                    feat_path = os.path.join(VIDEO_FEAT_DIR, row.video).replace(".mp4", ".pt")

                tensor_video = torch.load(feat_path).unsqueeze(0).to(DEVICE)
            else:
                tensor_video = get_tensor_video(images, preprocess_3d, "3D").unsqueeze(0).to(DEVICE)
            preds = model(tensor_video).squeeze(1).log_softmax(dim=1).argmax(dim=1)

            gloss_paths += save_glosses(images, pad_image, gloss_idx, temp_stride)
            for i in range(preds.size(0)):
                gloss = preds[i].item()
                Y.append(gloss)

            assert (len(Y) == len(gloss_paths))

            gloss_idx = len(Y)
            if SHOW_PROGRESS:
                pp.show(idx)

        if SHOW_PROGRESS:
            pp.end()

    Y_gloss = [vocab.idx2gloss[i] for i in Y]

    df = pd.DataFrame({"folder": gloss_paths, "gloss": Y_gloss, "gloss_idx": Y})

    L = df.shape[0]
    idxs = list(range(L))
    np.random.shuffle(idxs)
    df_train = df.iloc[idxs[:int(0.9 * L)]]
    df_val = df.iloc[idxs[int(0.9 * L):]]

    df_train.to_csv(os.path.join(ANNO_DIR, "gloss_train.csv"), index=None)
    df_val.to_csv(os.path.join(ANNO_DIR, "gloss_val.csv"), index=None)


if __name__ == "__main__":
    generate_gloss_dataset()
