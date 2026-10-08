from pathlib import Path

import torch


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


DATASET_ROOT = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "KolektorSDD2"
)


TRAIN_DIR = (
    DATASET_ROOT /
    "train"
)


TEST_DIR = (
    DATASET_ROOT /
    "test"
)


MODEL_DIR = (
    PROJECT_ROOT /
    "models"
)


OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs"
)


METRICS_DIR = (
    OUTPUT_DIR /
    "metrics"
)


PLOTS_DIR = (
    OUTPUT_DIR /
    "plots"
)


ERROR_DIR = (
    OUTPUT_DIR /
    "error_analysis"
)


for directory in [

    MODEL_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    ERROR_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Using device: {DEVICE}"
)


if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


IMAGE_HEIGHT = 256

IMAGE_WIDTH = 640

BATCH_SIZE = 16

NUM_WORKERS = 0

NUM_CLASSES = 2

CLASS_NAMES = [
    "Normal",
    "Defective"
]

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

VAL_SIZE = 0.20
