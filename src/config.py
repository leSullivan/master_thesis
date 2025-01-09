import os

SEED = 42

MODEL_TYPE = "cgan"
IMG_W = 768
IMG_H = 512
IMG_CH = 3
CROP = False

G_TYPE = "unet_128"
NGF = 64
N_DOWNSAMPLING_RES_NET = 2
N_DOWNSAMPLING_U_NET = 7

D_TYPE = "vagan"
NDF = 64
ND_LAYERS = 3
D_USE_SIGMOID = False

NUM_EPOCHS = 400
BATCH_SIZE = 2

LR = 0.00001
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 1e-2
ADAM_EPS = 1e-5
LAMBDA_GAN = 0.5
LAMBDA_L1 = 1
LAMBDA_PERCEPTUAL = 5
LAMBDA_CYCLE = 10

PROMPT_BG = "A gras landscape"
PROMPT_FENCE = "A gras landscape with a security mesh fence in the foreground"

NORM_TYPE = "instance"

NUM_WORKERS = 4

FENCE_IMG_PATH = os.path.join("imgs", "fences")
BG_IMG_PATH = os.path.join("imgs", "backgrounds")
RESULTS_IMG_PATH = os.path.join("imgs", "results")
TRAIN_IMG_PATH = os.path.join("imgs", "training_data")
