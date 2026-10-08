import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from fastapi.responses import HTMLResponse

from PIL import Image, UnidentifiedImageError

import io
import torch

from src.config import (
    DEVICE,
    MODEL_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CLASS_NAMES,
)

from src.predict import (
    predict_image,
    MODEL_PATH,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Visual Defect Detection API",
    description=(
        "Computer Vision API for industrial "
        "normal vs defective image classification."
    ),
    version="1.0.0",
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    model_exists = MODEL_PATH.exists()

    return {
        "status": "ok",
        "service": "Visual Defect Detection API",
        "device": str(DEVICE),
        "model_exists": model_exists,
        "model_path": str(MODEL_PATH),
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "EfficientNet-B0",
        "framework": "PyTorch",
        "task": "Industrial Defect Classification",
        "classes": CLASS_NAMES,
        "input_size": f"{IMAGE_HEIGHT} × {IMAGE_WIDTH}",
        "training": "Transfer Learning",
        "inference_device": str(DEVICE),
        "model_file": MODEL_PATH.name,
        "model_available": MODEL_PATH.exists(),
    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

async def run_prediction(file: UploadFile):

    if not file:
        raise HTTPException(
            status_code=400,
            detail="No image file was provided."
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/bmp",
        "image/webp",
        "image/tiff",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload JPG, PNG, BMP, WEBP or TIFF."
            )
        )

    try:

        contents = await file.read()

        if len(contents) == 0:

            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Limit upload size to 10 MB
        if len(contents) > 10 * 1024 * 1024:

            raise HTTPException(
                status_code=413,
                detail="Image file is too large. Maximum size is 10 MB."
            )

        image = Image.open(
            io.BytesIO(contents)
        )

        image.load()

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {str(e)}"
        )

    try:

        result = predict_image(image)

        result["filename"] = file.filename

        return result

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    return await run_prediction(file)


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...)
):

    return await run_prediction(file)


# ============================================================
# WEB INTERFACE
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Visual Defect Detection
</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top left,
            #1e293b,
            #0f172a 45%,
            #020617
        );

    color: #f8fafc;

    min-height: 100vh;
}

.container {

    width: min(1180px, 92%);

    margin: auto;
}

header {

    padding: 28px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.08);
}

.nav {

    display: flex;

    justify-content: space-between;

    align-items: center;
}

.logo {

    font-size: 22px;

    font-weight: 800;

    letter-spacing: -0.5px;
}

.logo span {

    color: #38bdf8;
}

.status {

    display: flex;

    align-items: center;

    gap: 8px;

    background:
        rgba(34,197,94,0.12);

    border:
        1px solid rgba(34,197,94,0.25);

    color: #86efac;

    padding: 8px 13px;

    border-radius: 999px;

    font-size: 13px;
}

.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;
}

.hero {

    padding: 55px 0 30px;
}

.hero h1 {

    margin: 0;

    font-size:
        clamp(36px, 5vw, 58px);

    line-height: 1.05;

    letter-spacing: -2px;
}

.hero h1 span {

    color: #38bdf8;
}

.hero p {

    max-width: 680px;

    color: #94a3b8;

    font-size: 17px;

    line-height: 1.7;

    margin-top: 20px;
}

.grid {

    display: grid;

    grid-template-columns:
        minmax(0, 1.5fr)
        minmax(300px, 0.8fr);

    gap: 24px;

    padding-bottom: 60px;
}

.card {

    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(255,255,255,0.09);

    border-radius: 20px;

    padding: 26px;

    backdrop-filter: blur(16px);

    box-shadow:
        0 20px 50px
        rgba(0,0,0,0.25);
}

.card h2 {

    margin-top: 0;

    font-size: 19px;
}

.upload-area {

    min-height: 360px;

    border:
        2px dashed #334155;

    border-radius: 16px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    padding: 25px;

    text-align: center;

    cursor: pointer;

    transition: 0.25s;
}

.upload-area:hover,
.upload-area.dragover {

    border-color: #38bdf8;

    background:
        rgba(56,189,248,0.05);
}

.upload-icon {

    font-size: 48px;

    margin-bottom: 12px;
}

.upload-area strong {

    font-size: 18px;
}

.upload-area p {

    color: #64748b;

    margin: 10px 0 20px;
}

input[type=file] {

    display: none;
}

.preview {

    max-width: 100%;

    max-height: 300px;

    border-radius: 12px;

    display: none;

    object-fit: contain;
}

.file-name {

    margin-top: 15px;

    color: #94a3b8;

    font-size: 13px;

    word-break: break-word;
}

button {

    width: 100%;

    border: 0;

    border-radius: 12px;

    padding: 15px;

    margin-top: 18px;

    font-size: 16px;

    font-weight: 700;

    cursor: pointer;

    background:
        linear-gradient(
            135deg,
            #0ea5e9,
            #2563eb
        );

    color: white;

    transition: 0.2s;
}

button:hover {

    transform: translateY(-1px);

    box-shadow:
        0 10px 30px
        rgba(14,165,233,0.25);
}

button:disabled {

    opacity: 0.5;

    cursor: not-allowed;

    transform: none;
}

.result {

    display: none;

    margin-top: 24px;

    border-radius: 16px;

    padding: 22px;

    border: 1px solid #334155;
}

.result.normal {

    background:
        rgba(34,197,94,0.08);

    border-color:
        rgba(34,197,94,0.3);
}

.result.defective {

    background:
        rgba(239,68,68,0.08);

    border-color:
        rgba(239,68,68,0.3);
}

.result-label {

    color: #94a3b8;

    font-size: 13px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.result-value {

    font-size: 36px;

    font-weight: 800;

    margin: 5px 0 20px;
}

.metric {

    margin-top: 14px;
}

.metric-row {

    display: flex;

    justify-content: space-between;

    margin-bottom: 6px;

    font-size: 13px;

    color: #cbd5e1;
}

.bar {

    height: 8px;

    border-radius: 20px;

    background: #1e293b;

    overflow: hidden;
}

.bar-fill {

    height: 100%;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            #0ea5e9,
            #38bdf8
        );

    transition: width 0.5s ease;
}

.model-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 12px;
}

.model-item {

    padding: 15px;

    background:
        rgba(255,255,255,0.035);

    border-radius: 12px;

    border: 1px solid
        rgba(255,255,255,0.05);
}

.model-item small {

    display: block;

    color: #64748b;

    margin-bottom: 5px;
}

.model-item strong {

    font-size: 14px;
}

.info {

    margin-top: 20px;

    color: #64748b;

    font-size: 13px;

    line-height: 1.6;
}

.error {

    display: none;

    margin-top: 18px;

    padding: 14px;

    border-radius: 10px;

    background:
        rgba(239,68,68,0.1);

    border:
        1px solid
        rgba(239,68,68,0.3);

    color: #fca5a5;

    font-size: 14px;
}

.loading {

    display: none;

    text-align: center;

    padding: 15px;

    color: #38bdf8;
}

footer {

    border-top:
        1px solid rgba(255,255,255,0.08);

    padding: 25px 0;

    color: #64748b;

    font-size: 13px;

    text-align: center;
}

@media(max-width: 850px) {

    .grid {

        grid-template-columns: 1fr;
    }

    .model-grid {

        grid-template-columns: 1fr 1fr;
    }
}

@media(max-width: 500px) {

    .model-grid {

        grid-template-columns: 1fr;
    }
}

</style>

</head>


<body>

<header>

<div class="container nav">

<div class="logo">
Vision<span>Defect</span>
</div>

<div class="status">

<div class="status-dot"></div>

API Online

</div>

</div>

</header>


<main class="container">

<section class="hero">

<h1>
Industrial Visual<br>
<span>Defect Detection</span>
</h1>

<p>
Upload an industrial product image and let the
deep learning model classify it as Normal or
Defective with confidence scores.
</p>

</section>


<section class="grid">


<!-- ======================================================
UPLOAD
======================================================= -->

<div class="card">

<h2>
Image Analysis
</h2>

<div
    class="upload-area"
    id="uploadArea"
>

<div
    class="upload-icon"
>
⌁
</div>

<strong>
Drop your image here
</strong>

<p>
or click to browse
</p>

<input
    type="file"
    id="fileInput"
    accept="image/*"
/>

<img
    id="preview"
    class="preview"
/>

<div
    id="fileName"
    class="file-name"
></div>

</div>


<button
    id="analyzeButton"
    disabled
>
Analyze Image
</button>


<div
    id="loading"
    class="loading"
>
Analyzing image...
</div>


<div
    id="error"
    class="error"
></div>


<div
    id="result"
    class="result"
>

<div class="result-label">
Prediction
</div>

<div
    id="prediction"
    class="result-value"
>
-
</div>


<div class="metric">

<div class="metric-row">

<span>
Confidence
</span>

<strong id="confidence">
0%
</strong>

</div>

<div class="bar">

<div
    id="confidenceBar"
    class="bar-fill"
    style="width:0%"
></div>

</div>

</div>


<div class="metric">

<div class="metric-row">

<span>
Normal Probability
</span>

<strong id="normalProbability">
0%
</strong>

</div>

<div class="bar">

<div
    id="normalBar"
    class="bar-fill"
    style="width:0%"
></div>

</div>

</div>


<div class="metric">

<div class="metric-row">

<span>
Defective Probability
</span>

<strong id="defectiveProbability">
0%
</strong>

</div>

<div class="bar">

<div
    id="defectiveBar"
    class="bar-fill"
    style="width:0%"
></div>

</div>

</div>

</div>

</div>


<!-- ======================================================
MODEL INFO
======================================================= -->

<div class="card">

<h2>
Model Information
</h2>

<div class="model-grid">

<div class="model-item">

<small>
Architecture
</small>

<strong id="modelName">
Loading...
</strong>

</div>


<div class="model-item">

<small>
Framework
</small>

<strong id="framework">
Loading...
</strong>

</div>


<div class="model-item">

<small>
Task
</small>

<strong id="task">
Loading...
</strong>

</div>


<div class="model-item">

<small>
Input
</small>

<strong id="inputSize">
Loading...
</strong>

</div>


<div class="model-item">

<small>
Classes
</small>

<strong id="classes">
Loading...
</strong>

</div>


<div class="model-item">

<small>
Inference
</small>

<strong id="device">
Loading...
</strong>

</div>

</div>


<div class="info">

<strong>
Deployment
</strong>

<p>
The trained EfficientNet-B0 model is loaded
from the application's <code>models</code> directory.
The application automatically uses CUDA when
available and CPU otherwise.
</p>

<strong>
Model Status
</strong>

<p id="modelStatus">
Checking model...
</p>

</div>

</div>

</section>

</main>


<footer>

Visual Defect Detection System ·
PyTorch · EfficientNet-B0

</footer>


<script>

const uploadArea =
    document.getElementById("uploadArea");

const fileInput =
    document.getElementById("fileInput");

const preview =
    document.getElementById("preview");

const fileName =
    document.getElementById("fileName");

const analyzeButton =
    document.getElementById("analyzeButton");

const loading =
    document.getElementById("loading");

const errorBox =
    document.getElementById("error");

const result =
    document.getElementById("result");


let selectedFile = null;


// ========================================================
// FILE SELECTION
// ========================================================

uploadArea.addEventListener(
    "click",
    () => fileInput.click()
);


fileInput.addEventListener(
    "change",
    event => {

        const file =
            event.target.files[0];

        handleFile(file);
    }
);


uploadArea.addEventListener(
    "dragover",
    event => {

        event.preventDefault();

        uploadArea.classList.add(
            "dragover"
        );
    }
);


uploadArea.addEventListener(
    "dragleave",
    () => {

        uploadArea.classList.remove(
            "dragover"
        );
    }
);


uploadArea.addEventListener(
    "drop",
    event => {

        event.preventDefault();

        uploadArea.classList.remove(
            "dragover"
        );

        const file =
            event.dataTransfer.files[0];

        handleFile(file);
    }
);


function handleFile(file) {

    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {

        showError(
            "Please select a valid image."
        );

        return;
    }

    selectedFile = file;

    const reader =
        new FileReader();

    reader.onload = event => {

        preview.src =
            event.target.result;

        preview.style.display =
            "block";

    };

    reader.readAsDataURL(file);

    fileName.textContent =
        file.name;

    analyzeButton.disabled =
        false;

    errorBox.style.display =
        "none";

    result.style.display =
        "none";
}


// ========================================================
// ANALYZE
// ========================================================

analyzeButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {
            return;
        }

        const formData =
            new FormData();

        formData.append(
            "file",
            selectedFile
        );

        loading.style.display =
            "block";

        analyzeButton.disabled =
            true;

        errorBox.style.display =
            "none";

        result.style.display =
            "none";

        try {

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const contentType =
                response.headers.get(
                    "content-type"
                ) || "";


            // Prevent:
            // Unexpected token '<'
            if (!contentType.includes(
                "application/json"
            )) {

                const text =
                    await response.text();

                throw new Error(
                    "Server returned a non-JSON response. " +
                    "HTTP " +
                    response.status +
                    ". " +
                    text.substring(0, 200)
                );
            }


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prediction failed."
                );
            }


            displayResult(data);

        }

        catch (error) {

            showError(
                error.message
            );

        }

        finally {

            loading.style.display =
                "none";

            analyzeButton.disabled =
                false;
        }

    }
);


// ========================================================
// DISPLAY RESULT
// ========================================================

function displayResult(data) {

    const prediction =
        data.prediction;

    const confidence =
        Number(data.confidence);

    const normal =
        Number(data.normal_probability);

    const defective =
        Number(data.defective_probability);


    document.getElementById(
        "prediction"
    ).textContent =
        prediction;


    document.getElementById(
        "confidence"
    ).textContent =
        confidence + "%";


    document.getElementById(
        "normalProbability"
    ).textContent =
        normal + "%";


    document.getElementById(
        "defectiveProbability"
    ).textContent =
        defective + "%";


    document.getElementById(
        "confidenceBar"
    ).style.width =
        confidence + "%";


    document.getElementById(
        "normalBar"
    ).style.width =
        normal + "%";


    document.getElementById(
        "defectiveBar"
    ).style.width =
        defective + "%";


    result.className =
        "result " +
        (
            prediction === "Defective"
                ? "defective"
                : "normal"
        );


    result.style.display =
        "block";
}


// ========================================================
// ERROR
// ========================================================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";
}


// ========================================================
// MODEL INFORMATION
// ========================================================

async function loadModelInfo() {

    try {

        const response =
            await fetch(
                "/model-info"
            );


        const data =
            await response.json();


        document.getElementById(
            "modelName"
        ).textContent =
            data.model;


        document.getElementById(
            "framework"
        ).textContent =
            data.framework;


        document.getElementById(
            "task"
        ).textContent =
            data.task;


        document.getElementById(
            "inputSize"
        ).textContent =
            data.input_size;


        document.getElementById(
            "classes"
        ).textContent =
            data.classes.join(
                " / "
            );


        document.getElementById(
            "device"
        ).textContent =
            data.inference_device;


        document.getElementById(
            "modelStatus"
        ).textContent =
            data.model_available
                ? "Trained model loaded and available."
                : "Model file is missing.";

    }

    catch (error) {

        document.getElementById(
            "modelStatus"
        ).textContent =
            "Could not retrieve model information.";
    }
}


loadModelInfo();

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return HTML_PAGE