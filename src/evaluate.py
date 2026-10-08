import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from src.config import (
    DEVICE,
    MODEL_DIR,
    METRICS_DIR,
    ERROR_DIR,
    CLASS_NAMES,
)

from src.dataset import (
    build_dataframe,
    assign_labels,
    KolektorDataset,
    get_val_transform,
)

from src.model import create_model

from torch.utils.data import DataLoader


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "best_model.pth"

model = create_model(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model = model.to(DEVICE)
model.eval()


# ============================================================
# TEST DATA
# ============================================================

from src.config import TEST_DIR

test_df = build_dataframe(
    TEST_DIR
)

test_df = assign_labels(
    test_df
)

test_dataset = KolektorDataset(
    test_df,
    transform=get_val_transform()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []


with torch.inference_mode():

    for images, labels, paths in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        all_paths.extend(paths)


y_true = np.array(
    all_labels
)

y_pred = np.array(
    all_predictions
)

y_prob = np.array(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

pr_auc = average_precision_score(
    y_true,
    y_prob
)

mcc = matthews_corrcoef(
    y_true,
    y_pred
)

kappa = cohen_kappa_score(
    y_true,
    y_pred
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

print(f"Accuracy:              {accuracy:.4f}")
print(f"Balanced Accuracy:     {balanced_accuracy:.4f}")
print(f"Precision:             {precision:.4f}")
print(f"Recall / Sensitivity:  {recall:.4f}")
print(f"Specificity:           {specificity:.4f}")
print(f"F1 Score:              {f1:.4f}")
print(f"ROC-AUC:               {roc_auc:.4f}")
print(f"PR-AUC:                {pr_auc:.4f}")
print(f"MCC:                   {mcc:.4f}")
print(f"Cohen Kappa:           {kappa:.4f}")

print("\nConfusion Matrix:")
print(
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "accuracy": float(accuracy),

    "balanced_accuracy":
        float(balanced_accuracy),

    "precision":
        float(precision),

    "recall_sensitivity":
        float(recall),

    "specificity":
        float(specificity),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "pr_auc":
        float(pr_auc),

    "matthews_correlation_coefficient":
        float(mcc),

    "cohen_kappa":
        float(kappa),

    "false_positive_rate":
        float(false_positive_rate),

    "false_negative_rate":
        float(false_negative_rate),

    "true_negatives":
        int(tn),

    "false_positives":
        int(fp),

    "false_negatives":
        int(fn),

    "true_positives":
        int(tp),

    "test_samples":
        int(len(y_true)),
}


with open(
    METRICS_DIR / "test_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    METRICS_DIR / "classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp],
    ],
    index=[
        "Actual Normal",
        "Actual Defective",
    ],
    columns=[
        "Predicted Normal",
        "Predicted Defective",
    ]
)

cm.to_csv(
    METRICS_DIR / "confusion_matrix.csv"
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "image_path":
        all_paths,

    "actual":
        y_true,

    "predicted":
        y_pred,

    "defective_probability":
        y_prob,

    "correct":
        y_true == y_pred,
})


predictions_df.to_csv(
    METRICS_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# FALSE POSITIVES
# ============================================================

fp_df = predictions_df[
    (predictions_df["actual"] == 0) &
    (predictions_df["predicted"] == 1)
]

fp_df.to_csv(
    ERROR_DIR / "false_positives.csv",
    index=False
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

fn_df = predictions_df[
    (predictions_df["actual"] == 1) &
    (predictions_df["predicted"] == 0)
]

fn_df.to_csv(
    ERROR_DIR / "false_negatives.csv",
    index=False
)


# ============================================================
# ROC CURVE DATA
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_prob
)

pd.DataFrame({

    "false_positive_rate":
        fpr,

    "true_positive_rate":
        tpr,

    "threshold":
        roc_thresholds,

}).to_csv(
    METRICS_DIR / "roc_curve.csv",
    index=False
)


# ============================================================
# PRECISION-RECALL CURVE DATA
# ============================================================

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_true,
    y_prob
)

pd.DataFrame({

    "precision":
        pr_precision,

    "recall":
        pr_recall,

}).to_csv(
    METRICS_DIR / "precision_recall_curve.csv",
    index=False
)


print("\nEvaluation files saved to:")
print(METRICS_DIR)

print("\nError analysis saved to:")
print(ERROR_DIR)