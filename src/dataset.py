import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import torch

from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision import transforms


from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    VAL_SIZE,
)


# =========================================================
# IMAGE EXTENSIONS
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# =========================================================
# FIND IMAGES
# =========================================================

def find_images(directory):

    return sorted(
        [
            p
            for p in directory.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


# =========================================================
# BUILD DATAFRAME
# =========================================================

def build_dataframe(directory):

    images = find_images(
        directory
    )

    if len(images) == 0:

        raise RuntimeError(
            f"No images found in {directory}"
        )

    return pd.DataFrame(
        {
            "image_path": [
                str(p)
                for p in images
            ]
        }
    )


# =========================================================
# POSSIBLE MASK NAMES
# =========================================================

def possible_mask_paths(
    image_path
):

    image_path = Path(
        image_path
    )

    stem = image_path.stem

    parent = image_path.parent

    candidates = [

        parent / f"{stem}_mask.png",

        parent / f"{stem}_mask.jpg",

        parent / f"{stem}_mask.bmp",

        parent / f"{stem}_label.png",

        parent / f"{stem}_label.jpg",

        parent / f"{stem}_seg.png",

        parent / f"{stem}_seg.jpg",

        parent / f"{stem}_GT.png",

        parent / f"{stem}_GT.jpg",

        parent / f"{stem}_gt.png",

        parent / f"{stem}_gt.jpg",

    ]

    return candidates


# =========================================================
# CHECK WHETHER MASK IS DEFECTIVE
# =========================================================

def mask_has_defect(
    mask_path
):

    try:

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            return False

        return bool(
            np.any(mask > 0)
        )

    except Exception:

        return False


# =========================================================
# FIND ASSOCIATED MASK
# =========================================================

def find_annotation(
    image_path
):

    image_path = Path(
        image_path
    )

    # Direct candidate paths
    for candidate in possible_mask_paths(
        image_path
    ):

        if candidate.exists():

            return candidate


    # Search nearby files having same stem
    parent = image_path.parent

    stem = image_path.stem.lower()

    for path in parent.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path == image_path:
            continue

        path_stem = path.stem.lower()

        if stem in path_stem:

            if any(
                keyword in path_stem
                for keyword in [
                    "mask",
                    "label",
                    "seg",
                    "gt",
                    "annotation",
                ]
            ):

                return path


    return None


# =========================================================
# ASSIGN LABELS
# =========================================================

def assign_labels(df):

    df = df.copy()

    labels = []

    annotations = []

    for image_path in df[
        "image_path"
    ]:

        annotation = find_annotation(
            image_path
        )

        annotations.append(
            str(annotation)
            if annotation
            else ""
        )


        if annotation is None:

            labels.append(0)

        else:

            defective = (
                mask_has_defect(
                    annotation
                )
            )

            labels.append(
                1 if defective else 0
            )


    df["annotation_path"] = (
        annotations
    )

    df["label"] = labels

    return df


# =========================================================
# SPLIT
# =========================================================

def create_split(
    df
):

    train_df, val_df = (
        train_test_split(
            df,
            test_size=VAL_SIZE,
            random_state=RANDOM_SEED,
            stratify=df["label"],
        )
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    return (
        train_df,
        val_df
    )


# =========================================================
# TRANSFORMS
# =========================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_train_transform():

    return transforms.Compose(

        [

            transforms.Resize(
                (
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=5
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_val_transform():

    return transforms.Compose(

        [

            transforms.Resize(
                (
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# =========================================================
# DATASET
# =========================================================

class KolektorDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(
        self,
        index
    ):

        row = self.dataframe.iloc[
            index
        ]

        image_path = row[
            "image_path"
        ]

        label = int(
            row["label"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            label,
            image_path,
        )


# =========================================================
# DATALOADERS
# =========================================================

def create_dataloaders(
    train_df,
    val_df
):

    train_dataset = (
        KolektorDataset(
            train_df,
            get_train_transform()
        )
    )

    val_dataset = (
        KolektorDataset(
            val_df,
            get_val_transform()
        )
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),
    )


    return (
        train_loader,
        val_loader
    )