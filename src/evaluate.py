import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.config import (
    DEVICE,
    MODEL_DIR,
    METRICS_DIR,
    ERROR_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.model import create_model

from src.dataset import (
    KolektorDataset,
    get_val_transform,
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    model = create_model(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_DIR
        / "best_model.pth",
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# =========================================================
# EVALUATE
# =========================================================

def evaluate(
    test_df
):

    model = load_model()


    dataset = KolektorDataset(
        test_df,
        transform=get_val_transform()
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


    all_labels = []

    all_predictions = []

    all_probabilities = []

    all_paths = []


    with torch.no_grad():

        for images, labels, paths in loader:

            images = images.to(
                DEVICE
            )


            outputs = model(
                images
            )


            probabilities = (
                torch.softmax(
                    outputs,
                    dim=1
                )
            )


            predictions = (
                torch.argmax(
                    probabilities,
                    dim=1
                )
            )


            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

            all_paths.extend(
                paths
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


    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )


    print("\nTEST RESULTS")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Normal",
                "Defective",
            ],
            zero_division=0,
        )
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual_Normal",
            "Actual_Defective",
        ],
        columns=[
            "Predicted_Normal",
            "Predicted_Defective",
        ],
    )


    cm_df.to_csv(
        METRICS_DIR
        / "confusion_matrix.csv"
    )


    # =====================================================
    # PREDICTIONS
    # =====================================================

    results = pd.DataFrame({

        "image_path":
            all_paths,

        "actual":
            all_labels,

        "predicted":
            all_predictions,

        "defective_probability":
            all_probabilities,
    })


    results["correct"] = (
        results["actual"]
        ==
        results["predicted"]
    )


    results.to_csv(
        METRICS_DIR
        / "test_predictions.csv",
        index=False,
    )


    # =====================================================
    # FALSE POSITIVES
    # =====================================================

    false_positives = (
        results[
            (
                results["actual"] == 0
            )
            &
            (
                results["predicted"] == 1
            )
        ]
    )


    false_positives.to_csv(
        ERROR_DIR
        / "false_positives.csv",
        index=False,
    )


    # =====================================================
    # FALSE NEGATIVES
    # =====================================================

    false_negatives = (
        results[
            (
                results["actual"] == 1
            )
            &
            (
                results["predicted"] == 0
            )
        ]
    )


    false_negatives.to_csv(
        ERROR_DIR
        / "false_negatives.csv",
        index=False,
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "results": results,
    }