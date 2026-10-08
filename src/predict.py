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


import torch

from PIL import Image

from torchvision import transforms


from src.config import (
    DEVICE,
    MODEL_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CLASS_NAMES,
)


from src.model import create_model


MODEL_PATH = (
    MODEL_DIR /
    "best_model.pth"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    ),

])


# ============================================================
# MODEL
# ============================================================

_model = None


def load_model():

    global _model


    if _model is not None:

        return _model


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )


    model = create_model(
        num_classes=len(
            CLASS_NAMES
        )
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict,
        strict=True
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    _model = model


    return _model


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    image: Image.Image
):

    model = load_model()


    if image is None:

        raise ValueError(
            "No image was provided."
        )


    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )


    tensor = transform(
        image
    )


    tensor = tensor.unsqueeze(
        0
    )


    tensor = tensor.to(
        DEVICE
    )


    with torch.inference_mode():

        outputs = model(
            tensor
        )


        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )
        )


    probabilities = probabilities[
        0
    ]


    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )


    prediction = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    normal_probability = float(
        probabilities[
            0
        ].item()
    )


    defective_probability = float(
        probabilities[
            1
        ].item()
    )


    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )


    return {

        "prediction":
            prediction,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "normal_probability":
            round(
                normal_probability * 100,
                2
            ),

        "defective_probability":
            round(
                defective_probability * 100,
                2
            ),

        "device":
            str(DEVICE),

        "model":
            "EfficientNet-B0",

    }

