import os

# GENERAL VARIABLES

PH_DIR = "path/to/RWTH-PHOENIX Weather 2014/directory"
KRSL_DIR = "/path/to/KRSL/directory"
PH_EVA_DIR = os.sep.join([PH_DIR, "evaluation"])
VARS_DIR = "/path/to/dir/where/code/variables/stored"
OPENPOSE_FOLDER = "/path/to/openpose/directory"

# SRC_MODE = "FULL"
SRC_MODE = "HAND"
SOURCE = "PH"
# SOURCE = "KRSL"

WEIGHTS_DIR = os.sep.join([VARS_DIR, SOURCE, SRC_MODE, "WEIGHTS"])
METRICS_DIR = os.sep.join([VARS_DIR, SOURCE, SRC_MODE, "METRICS"])

IMG_SIZE_2D = 224
IMG_SIZE_3D = 112

########################################################################################################################
# END TO END MODEL VARIABLES
DEVICE = "cuda:0"

END2END_MODEL_LOAD = True

if SOURCE == "PH":
    SRC_DIR = PH_DIR
    ANNO_DIR = os.sep.join([PH_DIR, "annotations"])
    VIDEOS_DIR = os.sep.join([PH_DIR, "features", "fullFrame-210x260px"])
    if SRC_MODE == "HAND":  VIDEOS_DIR = os.sep.join([PH_DIR, "features", "trackedRightHand-92x132px"])

else:
    SRC_DIR = KRSL_DIR
    ANNO_DIR = os.sep.join([KRSL_DIR, "annotation"])
    VIDEOS_DIR = os.path.join(KRSL_DIR, "videos")

USE_MP = True  # Use multi processing
USE_OVERFIT = False  # Load overfitted model as default
FEAT_OVERRIDE = True
USE_FEAT = False

# Feature Extractor models
# IMG_FEAT_MODEL = "densenet121"
# IMG_FEAT_MODEL = "googlenet"
# IMG_FEAT_MODEL = "pose"
IMG_FEAT_MODEL = "resnet{2+1}d"

VIDEO_FEAT_DIR = os.sep.join([SRC_DIR, "features", IMG_FEAT_MODEL, SRC_MODE])

TEMP_FUSION_TYPE = int(IMG_FEAT_MODEL == "resnet{2+1}d")  # 0 => 2D(feat ext and tempfusion), 1 => (2+1)D combined

if IMG_FEAT_MODEL in ["densenet121", "googlenet", "vgg-s", "resnet{2+1}d"]:
    IMG_FEAT_SIZE = 1024
elif IMG_FEAT_MODEL == "resnet18":
    IMG_FEAT_SIZE = 512
else:
    IMG_FEAT_SIZE = None

if IMG_FEAT_MODEL == "pose":
    USE_FEAT = True
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

if USE_FEAT:
    IMG_FEAT_MODEL = IMG_FEAT_MODEL + "(" + str(IMG_FEAT_SIZE) + ")"
else:
    img_size = IMG_SIZE_2D if TEMP_FUSION_TYPE == 0 else IMG_SIZE_3D
    IMG_FEAT_MODEL = IMG_FEAT_MODEL + "(img_" + str(img_size) + "x" + str(img_size) + ")"

END2END_N_EPOCHS = 100

if USE_FEAT:
    END2END_BATCH_SIZE = 64
else:
    END2END_BATCH_SIZE = 4

END2END_LR = 0.0001

# Augmentation constants
END2END_DATA_AUG_TEMP = True
END2END_DATA_AUG_FRAME = True
RANDOM_SKIP_TH = 0.3
DOWN_SAMPLE_FACTOR = 0.3

END2END_MODEL_PATH = os.sep.join([WEIGHTS_DIR, "END2END", IMG_FEAT_MODEL, "Val.pt"])
END2END_WER_PATH = os.sep.join([METRICS_DIR, "END2END_WER", IMG_FEAT_MODEL, "Val.txt"])

########################################################################################################################
# GR Model variables

# True to split end2end model, # False to use gr model trained before
USE_END2END_MODEL = False
IGNORE_BLANK = True

GLOSS_DATA_DIR = os.sep.join([SRC_DIR, "features", "gloss", SRC_MODE, "images"])

# For iterative training
GR_BATCH_SIZE = 32
GR_LR = 0.0001
GR_N_EPOCHS = 5

GR_END2END_MODEL_PATH = END2END_MODEL_PATH

if USE_OVERFIT:  GR_END2END_MODEL_PATH = END2END_MODEL_PATH.replace("Val", "Train")

GR_STF_MODEL_PATH = os.sep.join([WEIGHTS_DIR, "GR_STF", IMG_FEAT_MODEL + ".pt"])
GR_LOSS_PATH = os.sep.join([METRICS_DIR, "GR_LOSS", IMG_FEAT_MODEL + ".txt"])

if IGNORE_BLANK:
    GR_STF_MODEL_PATH = GR_STF_MODEL_PATH.replace(".pt", "_IGN_BLANK.pt")
    GR_LOSS_PATH = GR_LOSS_PATH.replace(".txt", "_IGN_BLANK.txt")

########################################################################################################################
# printing variables
SHOW_PROGRESS = True

SHOW_EXAMPLE = True

########################################################################################################################


