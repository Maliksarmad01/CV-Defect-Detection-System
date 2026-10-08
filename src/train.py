import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    DEVICE,
    TRAIN_DIR,
    MODEL_DIR,
    METRICS_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
)

from src.dataset import (
    build_dataframe,
    assign_labels,
    create_split,
    create_dataloaders,
)

from src.model import create_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# DATA
# ============================================================

df = build_dataframe(
    TRAIN_DIR
)

df = assign_labels(
    df
)

train_df, val_df = create_split(
    df
)

train_loader, val_loader = create_dataloaders(
    train_df,
    val_df
)


print(
    f"Training samples: {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)

print(
    "\nTraining distribution:"
)

print(
    train_df["label"].value_counts()
)


# ============================================================
# MODEL
# ============================================================

model = create_model(
    num_classes=2
)

model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_df["label"].values,
    minlength=2
)

class_weights = (
    len(train_df)
    /
    (2.0 * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print(
    "Class weights:",
    class_weights
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# AMP
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# HISTORY
# ============================================================

history = []

best_f1 = -1.0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    train_true = []
    train_pred = []

    for images, labels, _ in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        train_loss += (
            loss.item()
            *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_true.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        train_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    train_loss /= len(
        train_loader.dataset
    )


    train_accuracy = accuracy_score(
        train_true,
        train_pred
    )

    train_precision = precision_score(
        train_true,
        train_pred,
        zero_division=0
    )

    train_recall = recall_score(
        train_true,
        train_pred,
        zero_division=0
    )

    train_f1 = f1_score(
        train_true,
        train_pred,
        zero_division=0
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_true = []
    val_pred = []

    with torch.inference_mode():

        for images, labels, _ in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_true.extend(
                labels.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )


    val_loss /= len(
        val_loader.dataset
    )


    val_accuracy = accuracy_score(
        val_true,
        val_pred
    )

    val_precision = precision_score(
        val_true,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        val_true,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        val_true,
        val_pred,
        zero_division=0
    )


    scheduler.step(
        val_f1
    )


    current_lr = optimizer.param_groups[0]["lr"]


    row = {

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "train_precision":
            train_precision,

        "train_recall":
            train_recall,

        "train_f1":
            train_f1,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_precision":
            val_precision,

        "val_recall":
            val_recall,

        "val_f1":
            val_f1,

        "learning_rate":
            current_lr,
    }

    history.append(row)


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_f1:

        best_f1 = val_f1

        checkpoint = {

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_val_f1":
                best_f1,

            "val_accuracy":
                val_accuracy,

            "val_precision":
                val_precision,

            "val_recall":
                val_recall,
        }

        torch.save(
            checkpoint,
            MODEL_DIR / "best_model.pth"
        )

        print(
            "Best model saved."
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    METRICS_DIR / "training_history.csv",
    index=False
)


print(
    "\nTraining complete."
)

print(
    f"Best validation F1: {best_f1:.4f}"
)

print(
    f"Model: {MODEL_DIR / 'best_model.pth'}"
)