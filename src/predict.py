import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch

from PIL import Image

from torchvision import transforms

from src.config import (
    DEVICE,
    MODEL_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

from src.model import create_model


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = (
    MODEL_DIR
    / "best_model.pth"
)


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = create_model(
        num_classes=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
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
# TRANSFORM
# =========================================================

transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# =========================================================
# PREDICTION
# =========================================================

def predict_image(
    image: Image.Image
):

    model = load_model()

    image_tensor = transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = (
            torch.softmax(
                outputs,
                dim=1,
            )
        )


    normal_probability = (
        probabilities[0][0]
        .item()
    )

    defective_probability = (
        probabilities[0][1]
        .item()
    )


    if (
        defective_probability
        >= normal_probability
    ):

        prediction = "Defective"

        confidence = (
            defective_probability
            * 100
        )

    else:

        prediction = "Normal"

        confidence = (
            normal_probability
            * 100
        )


    return {

        "prediction":
            prediction,

        "confidence":
            confidence,

        "normal_probability":
            normal_probability,

        "defective_probability":
            defective_probability,
    }