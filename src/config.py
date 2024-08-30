import os
import torch

SEED = 42

MODEL_TYPE = "cgan"
IMG_W = 750
IMG_H = 500
IMG_CH = 3

GENERATOR_TYPE = "resnet"
NGF = 64
N_DOWNSAMPLING = 4

DISCRIMINATOR_TYPE = "basic"
NDF = 64
ND_LAYERS = 3

NUM_EPOCHS = 500
BATCH_SIZE = 10

LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 10


NORM_TYPE = "instance"

NUM_WORKERS = 4

FENCE_IMG_PATH = os.path.join("imgs", "fences")
BG_IMG_PATH = os.path.join("imgs", "backgrounds")
RESULTS_IMG_PATH = os.path.join("imgs", "results")
TRAIN_IMG_PATH = os.path.join("imgs", "training_data")
