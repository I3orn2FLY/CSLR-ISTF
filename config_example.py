import os

# GENERAL VARIABLES
PH_DIR = "path/to/RWTH-PHOENIX Weather 2014/directory"
KRSL_DIR = "/path/to/KRSL/directory"
VARS_DIR = "/path/to/dir/where/code/variables/stored"

# not required if you won't use openpose features
OPENPOSE_FOLDER = "/path/to/openpose/directory"

# the ".ctm" file for evaluating the model using phoenix script will be created in this folder
PH_EVA_DIR = os.sep.join([PH_DIR, "evaluation"])

DEVICE = "cuda:0"

SOURCE = "PH"
# SOURCE = "KRSL"

SRC_MODE = "FULL"
# SRC_MODE = "HAND"

vars_prefix = os.sep.join([VARS_DIR, SOURCE, SRC_MODE])

WEIGHTS_DIR = os.path.join(vars_prefix, "WEIGHTS")
ITER_VARS_DIR = os.path.join(vars_prefix, "ITERATIVE")
ITER_WEIGHTS = os.path.join(ITER_VARS_DIR, "WEIGHTS")
METRICS_DIR = os.path.join(vars_prefix, "METRICS")
GEN_DATA_DIR = os.path.join(vars_prefix, "GEN_DATA")

END2END_DATASETS_DIR = os.sep.join([GEN_DATA_DIR, "DATASETS", "END2END"])
GR_DATASET_DIR = os.sep.join([GEN_DATA_DIR, "DATASETS", "GR"])
GR_ANNO_DIR = os.path.join(GR_DATASET_DIR, "annotation")
GR_VIDEOS_DIR = os.path.join(GEN_DATA_DIR, "GR_VIDEOS")

IMG_SIZE_2D = 224
IMG_SIZE_2Plus1D = 112

########################################################################################################################
# END TO END MODEL VARIABLES
END2END_MODEL_LOAD = False

if SOURCE == "PH":
    SRC_DIR = PH_DIR
    ANNO_DIR = os.sep.join([PH_DIR, "annotations"])
    VIDEOS_DIR = os.path.join(PH_DIR, "features")
    if SRC_MODE == "FULL":
        VIDEOS_DIR = os.path.join(VIDEOS_DIR, "fullFrame-210x260px")
    else:
        VIDEOS_DIR = os.path.join(VIDEOS_DIR, "trackedRightHand-92x132px")
else:
    SRC_DIR = KRSL_DIR
    ANNO_DIR = os.sep.join([KRSL_DIR, "annotation"])
    VIDEOS_DIR = os.path.join(KRSL_DIR, "videos")

FEAT_OVERRIDE = True
USE_STF_FEAT = True
USE_IMG_FEAT = True

# Spatio temporal Feature Extractor models
STF_MODEL = "densenet121"
# STF_MODEL = "googlenet"
# STF_MODEL = "pose"
# STF_MODEL = "resnet{2+1}d"

STF_FEAT_DIR = os.sep.join([GEN_DATA_DIR, "STF_FEATS", STF_MODEL])

STF_TYPE = int(STF_MODEL == "resnet{2+1}d")  # 0 => 2D(feat ext and temp fusion), 1 => (2+1)D combined

if STF_MODEL in ["densenet121", "googlenet", "vgg-s", "resnet{2+1}d"]:
    IMG_FEAT_SIZE = 1024
elif STF_MODEL == "resnet18":
    IMG_FEAT_SIZE = 512
elif STF_MODEL == "pose":
    USE_STF_FEAT = True
    POSE_BODY = True
    POSE_HANDS = True
    POSE_FACE = False
    IMG_FEAT_SIZE = 0
    if POSE_BODY:  IMG_FEAT_SIZE += 12 * 2
    if POSE_HANDS: IMG_FEAT_SIZE += 42 * 2
    if POSE_FACE:  IMG_FEAT_SIZE += 70 * 2

    POSE_AUG_NOISE_HANDFACE = 0.01
    POSE_AUG_NOISE_BODY = 0.02
    POSE_AUG_OFFSET = 0
else:
    IMG_FEAT_SIZE = None
    print("Wrong STF model")
    exit(0)

if USE_STF_FEAT:
    FEAT_TYPE = "feat_" + str(IMG_FEAT_SIZE)
else:
    img_size = IMG_SIZE_2D if STF_TYPE == 0 else IMG_SIZE_2Plus1D
    FEAT_TYPE = "img_" + str(img_size) + "x" + str(img_size)

END2END_N_EPOCHS = 100

END2END_STF_BATCH_SIZE = 32
END2END_RAW_BATCH_SIZE = 4

END2END_LR = 0.00005

# Augmentation constants
END2END_DATA_AUG_TEMP = True
END2END_DATA_AUG_FRAME = True
RANDOM_SKIP_TH = 0.3
DOWN_SAMPLE_FACTOR = 0.3
########################################################################################################################
# GR Train Variables
GR_BATCH_SIZE = 32
GR_LR = 0.00005
GR_N_EPOCHS = 10
########################################################################################################################
N_ITER = 6
END2END_STOP_LIMIT = 10
########################################################################################################################
load_crit = "val"
# load_crit = "train"

STF_MODEL_PATH = os.sep.join([WEIGHTS_DIR, STF_MODEL, str(IMG_FEAT_SIZE), "STF_" + load_crit + ".pt"])
SEQ2SEQ_MODEL_PATH = os.sep.join([WEIGHTS_DIR, STF_MODEL, str(IMG_FEAT_SIZE), "SEQ2SEQ_" + load_crit + ".pt"])
END2END_WER_PATH = os.sep.join([METRICS_DIR, STF_MODEL, str(IMG_FEAT_SIZE), "END2END_WER_" + load_crit + ".txt"])
GR_LOSS_PATH = os.sep.join([METRICS_DIR, STF_MODEL, "GR_LOSS.txt"])

########################################################################################################################
# printing variables
SHOW_PROGRESS = True

SHOW_EXAMPLE = True

########################################################################################################################


# Guideline to run
#