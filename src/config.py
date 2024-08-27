import os
import torch

SEED = 42

IMG_W = 750
IMG_H = 500
IMG_CH = 3

NUM_EPOCHS = 2000
LR = 0.0002
EARLY_STOPPING_PATIENCE = 20
BETA1 = 0.5
BETA2 = 0.999
BATCH_SIZE = 32
NUM_WORKERS = 4

FENCE_IMG_PATH = os.path.join("imgs", "fences")
BG_IMG_PATH = os.path.join("imgs", "backgrounds")
RESULTS_IMG_PATH = os.path.join("imgs", "results")
TRAIN_IMG_PATH = os.path.join("imgs", "training_data")
