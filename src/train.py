import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import torch

import torch.nn as nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)

from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_DIR,
    METRICS_DIR,
)

from src.model import (
    create_model
)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):

    model.train()

    total_loss = 0

    all_labels = []

    all_predictions = []


    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )


    for batch in progress:

        images = batch[0]

        labels = batch[1]


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
            device_type="cuda",
            enabled=(
                DEVICE.type == "cuda"
            ),
        ):

            outputs = model(
                images
            )

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


        total_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = (
            torch.argmax(
                outputs,
                dim=1
            )
        )


        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    average_loss = (
        total_loss
        / len(loader.dataset)
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "f1": f1,
    }


# =========================================================
# VALIDATION
# =========================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for batch in loader:

            images = batch[0]

            labels = batch[1]


            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            with torch.amp.autocast(
                device_type="cuda",
                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            total_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )


            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


    average_loss = (
        total_loss
        / len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )


    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )


    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    train_loader,
    val_loader,
    train_labels,
):

    model = create_model(
        num_classes=2
    )


    model = model.to(
        DEVICE
    )


    # -----------------------------------------------------
    # CLASS WEIGHTS
    # -----------------------------------------------------

    class_counts = np.bincount(
        train_labels,
        minlength=2
    )


    class_weights = (
        len(train_labels)
        /
        (
            2
            * class_counts
        )
    )


    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )


    print(
        "Class counts:",
        class_counts
    )

    print(
        "Class weights:",
        class_weights
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    )


    best_f1 = -1

    history = []


    # =====================================================
    # EPOCHS
    # =====================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\nEpoch "
            f"{epoch}/{EPOCHS}"
        )


        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
            )
        )


        val_metrics = (
            validate(
                model,
                val_loader,
                criterion,
            )
        )


        scheduler.step(
            val_metrics["f1"]
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        print(
            f"Train Loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Train Acc: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"Train F1: "
            f"{train_metrics['f1']:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Acc: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Precision: "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall: "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Val F1: "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )


        history.append({

            "epoch": epoch,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_f1":
                train_metrics["f1"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],

            "learning_rate":
                current_lr,
        })


        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        if (
            val_metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                val_metrics["f1"]
            )


            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "num_classes": 2,

                "class_names": [
                    "Normal",
                    "Defective",
                ],

                "image_size": (
                    256,
                    640
                ),

                "best_val_f1":
                    best_f1,

                "epoch":
                    epoch,

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_metrics":
                    val_metrics,
            }


            torch.save(
                checkpoint,
                MODEL_DIR
                / "best_model.pth"
            )


            print(
                "✓ Best model saved."
            )


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        METRICS_DIR
        / "training_history.csv",
        index=False,
    )


    return (
        model,
        history_df
    )


if __name__ == "__main__":

    print(
        "Training module loaded."
    )

    print(
        "Run training from "
        "02_training.ipynb"
    )