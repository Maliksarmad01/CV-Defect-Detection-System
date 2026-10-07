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

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Defect Detection</title>
<style>
:root {
    --bg: #e9edf1;
    --panel: #ffffff;
    --ink: #17202a;
    --muted: #5d6b7a;
    --line: #d3dae1;
    --accent: #1d4ed8;
    --ok: #15803d;
    --ok-bg: #e6f4ea;
    --bad: #b91c1c;
    --bad-bg: #fdeaea;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #11161c;
        --panel: #1a222b;
        --ink: #e8edf2;
        --muted: #93a1b0;
        --line: #2b3642;
        --accent: #6b9bff;
        --ok: #4ade80;
        --ok-bg: #12301f;
        --bad: #f87171;
        --bad-bg: #3a1717;
    }
}
* { box-sizing: border-box; }
body {
    margin: 0;
    min-height: 100vh;
    background: var(--bg);
    color: var(--ink);
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
    padding: 32px 20px;
    display: flex;
    justify-content: center;
}
.app { width: 100%; max-width: 980px; }
header { margin-bottom: 24px; }
h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: -0.02em; }
header p { margin: 0; color: var(--muted); max-width: 60ch; line-height: 1.5; }

.grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }
.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 20px;
}
.panel h2 { margin: 0 0 14px; font-size: 16px; }

/* Drop zone / preview */
.stage {
    position: relative;
    aspect-ratio: 4 / 3;
    border: 2px dashed var(--line);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    overflow: hidden;
    cursor: pointer;
    transition: border-color .15s, background .15s;
}
.stage:hover, .stage.drag, .stage:focus-visible {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 6%, transparent);
    outline: none;
}
.stage.has-image { border-style: solid; cursor: default; background: #0b0f14; }
.hint { color: var(--muted); padding: 20px; line-height: 1.5; }
.hint strong { display: block; color: var(--ink); font-size: 17px; margin-bottom: 4px; }
.stage img { width: 100%; height: 100%; object-fit: contain; display: none; }
.stage.has-image img { display: block; }
.stage.has-image .hint { display: none; }

/* Scan line while analyzing */
.scan {
    position: absolute;
    left: 0; right: 0; top: 0;
    height: 3px;
    background: var(--accent);
    box-shadow: 0 0 18px 4px var(--accent);
    display: none;
}
.stage.scanning .scan { display: block; animation: sweep 1.4s ease-in-out infinite alternate; }
@keyframes sweep { from { top: 0; } to { top: calc(100% - 3px); } }
@media (prefers-reduced-motion: reduce) {
    .stage.scanning .scan { animation: none; top: 50%; }
}

.file-meta { margin: 10px 0 0; font-size: 13px; color: var(--muted); min-height: 18px; word-break: break-all; }
.actions { display: flex; gap: 10px; margin-top: 14px; }
button {
    font: inherit;
    border-radius: 8px;
    padding: 12px 16px;
    cursor: pointer;
    border: 1px solid var(--line);
    background: transparent;
    color: var(--ink);
}
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button.primary {
    flex: 1;
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
}
@media (prefers-color-scheme: dark) { button.primary { color: #0b1220; } }
button:disabled { opacity: .45; cursor: not-allowed; }

/* Result */
.empty { color: var(--muted); line-height: 1.5; }
#result { display: none; }
.verdict {
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 18px;
}
.verdict.ok { background: var(--ok-bg); color: var(--ok); }
.verdict.bad { background: var(--bad-bg); color: var(--bad); }
.verdict .label { font-size: 26px; font-weight: 700; margin: 0; }
.verdict .conf { margin: 4px 0 0; font-size: 15px; }

.split {
    display: flex;
    height: 14px;
    border-radius: 7px;
    overflow: hidden;
    background: var(--line);
}
.split .n { background: var(--ok); width: 50%; transition: width .5s ease; }
.split .d { background: var(--bad); width: 50%; transition: width .5s ease; }
.legend { display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px; }
.legend span { display: block; color: var(--muted); font-size: 13px; }
.legend b { font-size: 20px; }
.legend .l b { color: var(--ok); }
.legend .r { text-align: right; }
.legend .r b { color: var(--bad); }

#error {
    display: none;
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 8px;
    background: var(--bad-bg);
    color: var(--bad);
    font-size: 14px;
    line-height: 1.4;
}
footer { margin-top: 22px; color: var(--muted); font-size: 13px; }

@media (max-width: 760px) {
    .grid { grid-template-columns: 1fr; }
    h1 { font-size: 25px; }
}
</style>
</head>
<body>
<div class="app">
    <header>
        <h1>Visual Defect Detection</h1>
        <p>Upload a photo of a product. The model classifies it as normal or defective and shows how sure it is.</p>
    </header>

    <div class="grid">
        <section class="panel" aria-labelledby="img-h">
            <h2 id="img-h">Image</h2>
            <div class="stage" id="stage" tabindex="0" role="button"
                 aria-label="Choose or drop an image">
                <div class="hint">
                    <strong>Drop an image here</strong>
                    or click to browse, or paste from your clipboard. JPG, PNG or WebP.
                </div>
                <img id="preview" alt="Selected product">
                <div class="scan"></div>
            </div>
            <input id="imageInput" type="file" accept="image/*" hidden>
            <p class="file-meta" id="fileMeta"></p>
            <div class="actions">
                <button class="primary" id="analyzeButton" disabled>Analyze image</button>
                <button id="clearButton" disabled>Clear</button>
            </div>
            <div id="error" role="alert"></div>
        </section>

        <section class="panel" aria-labelledby="res-h" aria-live="polite">
            <h2 id="res-h">Result</h2>
            <p class="empty" id="empty">Results appear here after you analyze an image.</p>
            <div id="result">
                <div class="verdict" id="verdict">
                    <p class="label" id="prediction"></p>
                    <p class="conf" id="confidence"></p>
                </div>
                <div class="split" aria-hidden="true">
                    <div class="n" id="barN"></div>
                    <div class="d" id="barD"></div>
                </div>
                <div class="legend">
                    <div class="l"><span>Normal</span><b id="normalProbability"></b></div>
                    <div class="r"><span>Defective</span><b id="defectiveProbability"></b></div>
                </div>
            </div>
        </section>
    </div>

    <footer>EfficientNet-B0 transfer learning &middot; PyTorch &middot; FastAPI</footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const stage = $("stage"), input = $("imageInput"), preview = $("preview");
const analyzeBtn = $("analyzeButton"), clearBtn = $("clearButton");
const MAX_MB = 10;
let currentFile = null;

function showError(msg) {
    $("error").textContent = msg;
    $("error").style.display = "block";
}
function hideError() { $("error").style.display = "none"; }

function setFile(file) {
    hideError();
    if (!file) return;
    if (!file.type.startsWith("image/")) {
        showError("That file is not an image. Choose a JPG, PNG or WebP file.");
        return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
        showError("The image is larger than " + MAX_MB + " MB. Choose a smaller file.");
        return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        stage.classList.add("has-image");
    };
    reader.readAsDataURL(file);
    $("fileMeta").textContent = file.name + " (" + (file.size / 1024).toFixed(0) + " KB)";
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
    $("result").style.display = "none";
    $("empty").style.display = "block";
}

function clearAll() {
    currentFile = null;
    input.value = "";
    preview.removeAttribute("src");
    stage.classList.remove("has-image", "scanning");
    $("fileMeta").textContent = "";
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
    $("result").style.display = "none";
    $("empty").style.display = "block";
    hideError();
}

stage.addEventListener("click", () => { if (!currentFile) input.click(); });
stage.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !currentFile) {
        e.preventDefault();
        input.click();
    }
});
input.addEventListener("change", () => setFile(input.files[0]));
clearBtn.addEventListener("click", clearAll);
analyzeBtn.addEventListener("click", analyzeImage);

["dragenter", "dragover"].forEach((t) =>
    stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) =>
    stage.addEventListener(t, (e) => { e.preventDefault(); stage.classList.remove("drag"); }));
stage.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

document.addEventListener("paste", (e) => {
    const items = e.clipboardData ? e.clipboardData.files : [];
    if (items.length) setFile(items[0]);
});

async function analyzeImage() {
    if (!currentFile) { showError("Choose an image first."); return; }
    hideError();
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";
    stage.classList.add("scanning");

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction failed. Try again.");

        const bad = String(data.prediction).toLowerCase().includes("defect");
        $("verdict").className = "verdict " + (bad ? "bad" : "ok");
        $("prediction").textContent = data.prediction;
        $("confidence").textContent = "Confidence: " + data.confidence + "%";

        const n = data.normal_probability * 100;
        const d = data.defective_probability * 100;
        $("normalProbability").textContent = n.toFixed(2) + "%";
        $("defectiveProbability").textContent = d.toFixed(2) + "%";
        $("barN").style.width = n + "%";
        $("barD").style.width = d + "%";

        $("empty").style.display = "none";
        $("result").style.display = "block";
    } catch (err) {
        showError(err.message);
    } finally {
        stage.classList.remove("scanning");
        analyzeBtn.textContent = "Analyze image";
        analyzeBtn.disabled = false;
    }
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