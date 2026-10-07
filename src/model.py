import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


def create_model(
    num_classes=2
):

    weights = (
        EfficientNet_B0_Weights.DEFAULT
    )

    model = efficientnet_b0(
        weights=weights
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(

        nn.Dropout(
            p=0.3
        ),

        nn.Linear(
            input_features,
            num_classes
        )
    )

    return model