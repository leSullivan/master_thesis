import os
import torch

SEED = 42

MODEL_TYPE = "cgan"
IMG_W = 768
IMG_H = 512
IMG_CH = 3

G_TYPE = "resnet-6"
NGF = 64
N_DOWNSAMPLING_RES_NET = 4
N_DOWNSAMPLING_U_NET = 6

D_TYPE = "basic"
NDF = 64
ND_LAYERS = 3
D_USE_SIGMOID = False

NUM_EPOCHS = 400
BATCH_SIZE = 5

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_GAN = 0.5
LAMBDA_L1 = 1
LAMBDA_PERCEPTUAL = 5
LAMBDA_CYCLE = 10

PROMPT_BG = "Please provide the path to the background image"
PROMPT_FENCE = "Please provide the path to the fence image"

NORM_TYPE = "instance"

NUM_WORKERS = 4

FENCE_IMG_PATH = os.path.join("imgs", "fences")
BG_IMG_PATH = os.path.join("imgs", "backgrounds")
RESULTS_IMG_PATH = os.path.join("imgs", "results")
TRAIN_IMG_PATH = os.path.join("imgs", "training_data")
