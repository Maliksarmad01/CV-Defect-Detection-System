import sys
from pathlib import Path

import io

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from fastapi.responses import (
    HTMLResponse,
)

from src.predict import (
    predict_image,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(

    title="Visual Defect Detection API",

    description=(
        "AI-powered industrial "
        "surface defect classification"
    ),

    version="1.0.0",
)


# =========================================================
# WEB PAGE
# =========================================================

HTML_PAGE = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>
Visual Defect Detection
</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    min-height: 100vh;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef2f7,
            #dfe6ee
        );

    color: #1f2937;

    display: flex;

    align-items: center;

    justify-content: center;

    padding: 30px;
}


.container {

    width: 100%;

    max-width: 850px;

    background: white;

    border-radius: 22px;

    padding: 40px;

    box-shadow:
        0 20px 60px
        rgba(0,0,0,0.12);
}


.header {

    text-align: center;

    margin-bottom: 35px;
}


.header h1 {

    margin: 0;

    font-size: 36px;
}


.header p {

    color: #6b7280;

    margin-top: 10px;

    font-size: 16px;
}


.upload-area {

    border:
        2px dashed #9ca3af;

    border-radius: 16px;

    padding: 35px;

    text-align: center;

    transition: 0.2s;

    cursor: pointer;
}


.upload-area:hover {

    border-color: #111827;

    background:
        #f9fafb;
}


.upload-area input {

    margin-top: 20px;

}


.preview-container {

    margin-top: 25px;

    text-align: center;

    display: none;
}


.preview {

    max-width: 100%;

    max-height: 350px;

    border-radius: 12px;

    box-shadow:
        0 8px 25px
        rgba(0,0,0,0.12);
}


.analyze-btn {

    width: 100%;

    margin-top: 25px;

    padding: 16px;

    border: none;

    border-radius: 10px;

    background:
        #111827;

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;
}


.analyze-btn:hover {

    background:
        #374151;
}


.analyze-btn:disabled {

    background:
        #9ca3af;

    cursor: not-allowed;
}


.result {

    display: none;

    margin-top: 30px;

    padding: 30px;

    border-radius: 16px;

    background:
        #f3f4f6;

    text-align: center;
}


.prediction {

    font-size: 32px;

    font-weight: bold;

    margin-bottom: 10px;
}


.confidence {

    font-size: 20px;

    margin-bottom: 25px;
}


.probability-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 15px;
}


.probability-card {

    padding: 20px;

    background: white;

    border-radius: 12px;
}


.probability-card h3 {

    margin-top: 0;

    font-size: 15px;

    color: #6b7280;
}


.probability-value {

    font-size: 24px;

    font-weight: bold;
}


.error {

    display: none;

    margin-top: 20px;

    padding: 15px;

    border-radius: 10px;

    background: #fee2e2;

    color: #991b1b;

    text-align: center;
}


.loading {

    display: none;

    text-align: center;

    margin-top: 20px;

    color: #6b7280;
}


.footer {

    text-align: center;

    color: #9ca3af;

    font-size: 13px;

    margin-top: 30px;
}


@media (max-width: 600px) {

    .container {

        padding: 25px;
    }

    .header h1 {

        font-size: 28px;
    }

    .probability-grid {

        grid-template-columns:
            1fr;
    }

}

</style>

</head>


<body>


<div class="container">


<div class="header">

<h1>
Visual Defect Detection
</h1>

<p>
Upload an industrial product image
to classify it as Normal or Defective.
</p>

</div>


<div
    class="upload-area"
    onclick="document
        .getElementById('imageInput')
        .click()"
>

<strong>
Choose an image
</strong>

<br>

<span>
JPG, JPEG, PNG supported
</span>


<input
    id="imageInput"
    type="file"
    accept="image/*"
    hidden
>


</div>


<div
    class="preview-container"
    id="previewContainer"
>

<img
    id="preview"
    class="preview"
>

</div>


<button
    id="analyzeButton"
    class="analyze-btn"
    onclick="analyzeImage()"
    disabled
>

Analyze Image

</button>


<div
    id="loading"
    class="loading"
>

Analyzing image with the trained model...

</div>


<div
    id="error"
    class="error"
>
</div>


<div
    id="result"
    class="result"
>


<div
    id="prediction"
    class="prediction"
>
</div>


<div
    id="confidence"
    class="confidence"
>
</div>


<div class="probability-grid">


<div class="probability-card">

<h3>
Normal Probability
</h3>

<div
    id="normalProbability"
    class="probability-value"
>
</div>

</div>


<div class="probability-card">

<h3>
Defective Probability
</h3>

<div
    id="defectiveProbability"
    class="probability-value"
>
</div>

</div>


</div>


</div>


<div class="footer">

EfficientNet-B0 Transfer Learning |
PyTorch | FastAPI

</div>


</div>


<script>


const imageInput =
    document.getElementById(
        "imageInput"
    );


const preview =
    document.getElementById(
        "preview"
    );


const previewContainer =
    document.getElementById(
        "previewContainer"
    );


const analyzeButton =
    document.getElementById(
        "analyzeButton"
    );


const result =
    document.getElementById(
        "result"
    );


const error =
    document.getElementById(
        "error"
    );


const loading =
    document.getElementById(
        "loading"
    );


imageInput.addEventListener(
    "change",
    function() {

        const file =
            imageInput.files[0];

        if (!file) {

            analyzeButton.disabled =
                true;

            return;
        }


        const reader =
            new FileReader();


        reader.onload =
            function(event) {

                preview.src =
                    event.target.result;

                previewContainer.style.display =
                    "block";

                analyzeButton.disabled =
                    false;

                result.style.display =
                    "none";

                error.style.display =
                    "none";
            };


        reader.readAsDataURL(
            file
        );
    }
);


async function analyzeImage() {

    const file =
        imageInput.files[0];


    if (!file) {

        showError(
            "Please select an image."
        );

        return;
    }


    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    analyzeButton.disabled =
        true;


    loading.style.display =
        "block";


    result.style.display =
        "none";


    error.style.display =
        "none";


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );
        }


        document.getElementById(
            "prediction"
        ).textContent =
            data.prediction;


        document.getElementById(
            "confidence"
        ).textContent =
            "Confidence: "
            +
            data.confidence
            +
            "%";


        document.getElementById(
            "normalProbability"
        ).textContent =
            (
                data.normal_probability
                * 100
            ).toFixed(2)
            +
            "%";


        document.getElementById(
            "defectiveProbability"
        ).textContent =
            (
                data.defective_probability
                * 100
            ).toFixed(2)
            +
            "%";


        result.style.display =
            "block";

    }


    catch (err) {

        showError(
            err.message
        );

    }


    finally {

        loading.style.display =
            "none";

        analyzeButton.disabled =
            false;
    }
}


function showError(
    message
) {

    error.textContent =
        message;

    error.style.display =
        "block";
}

</script>


</body>

</html>

"""


# =========================================================
# HOME
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "service":
            "visual-defect-detection",

    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not determine "
                "file type."
            )
        )


    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload "
                "an image file."
            )
        )


    contents = await file.read()


    if len(contents) == 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is empty."
            )
        )


    try:

        image = Image.open(
            io.BytesIO(
                contents
            )
        ).convert(
            "RGB"
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid or corrupted "
                "image."
            )
        )


    try:

        result = predict_image(
            image
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )