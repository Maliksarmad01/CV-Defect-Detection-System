import sys
import sqlite3
import hashlib
import secrets
import io
import os
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request,
    Form,
)
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    JSONResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR

DATABASE_PATH = PROJECT_ROOT / "users.db"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ML IMPORTS
# ============================================================

from src.predict import predict_image
from src.config import DEVICE, MODEL_DIR


MODEL_PATH = MODEL_DIR / "best_model.pth"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Visual Defect Detection System",
    description="AI-powered visual defect detection API",
    version="1.0.0",
)


# ============================================================
# SESSION
# ============================================================

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "development-secret-key-change-in-production",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=60 * 60 * 24 * 7,
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_db()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    connection.close()


initialize_database()


# ============================================================
# PASSWORD SECURITY
# ============================================================

def hash_password(password: str) -> str:

    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )

    return (
        salt.hex()
        + ":"
        + password_hash.hex()
    )


def verify_password(
    password: str,
    stored_hash: str,
) -> bool:

    try:

        salt_hex, hash_hex = (
            stored_hash.split(":")
        )

        salt = bytes.fromhex(
            salt_hex
        )

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100_000,
        )

        return secrets.compare_digest(
            password_hash.hex(),
            hash_hex,
        )

    except Exception:

        return False


# ============================================================
# AUTH HELPERS
# ============================================================

def get_current_user(request: Request):

    user_id = request.session.get(
        "user_id"
    )

    if not user_id:
        return None

    connection = get_db()

    user = connection.execute(
        """
        SELECT id, name, email
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()

    connection.close()

    return user


def require_user(request: Request):

    user = get_current_user(request)

    if user is None:

        return None

    return user


# ============================================================
# LANDING PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def landing_page():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ============================================================
# LOGIN PAGE
# ============================================================

@app.get(
    "/login",
    response_class=HTMLResponse,
)
async def login_page():

    return FileResponse(
        FRONTEND_DIR / "login.html"
    )


# ============================================================
# SIGNUP PAGE
# ============================================================

@app.get(
    "/signup",
    response_class=HTMLResponse,
)
async def signup_page():

    return FileResponse(
        FRONTEND_DIR / "signup.html"
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.get(
    "/dashboard",
    response_class=HTMLResponse,
)
async def dashboard_page(
    request: Request,
):

    user = require_user(request)

    if user is None:

        return RedirectResponse(
            "/login",
            status_code=303,
        )

    return FileResponse(
        FRONTEND_DIR / "dashboard.html"
    )


# ============================================================
# SIGNUP API
# ============================================================

@app.post("/api/signup")
async def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):

    name = name.strip()

    email = email.strip().lower()


    if len(name) < 2:

        raise HTTPException(
            status_code=400,
            detail="Name must contain at least 2 characters.",
        )


    if len(email) < 5:

        raise HTTPException(
            status_code=400,
            detail="Please enter a valid email address.",
        )


    if len(password) < 6:

        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters.",
        )


    connection = get_db()


    existing_user = connection.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (email,),
    ).fetchone()


    if existing_user:

        connection.close()

        raise HTTPException(
            status_code=400,
            detail="An account with this email already exists.",
        )


    password_hash = hash_password(
        password
    )


    cursor = connection.execute(
        """
        INSERT INTO users
        (
            name,
            email,
            password_hash
        )
        VALUES (?, ?, ?)
        """,
        (
            name,
            email,
            password_hash,
        ),
    )


    connection.commit()


    user_id = cursor.lastrowid


    connection.close()


    request.session["user_id"] = user_id


    return {
        "success": True,
        "message": "Account created successfully.",
        "redirect": "/dashboard",
    }


# ============================================================
# LOGIN API
# ============================================================

@app.post("/api/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):

    email = email.strip().lower()


    connection = get_db()


    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE email = ?
        """,
        (email,),
    ).fetchone()


    connection.close()


    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )


    if not verify_password(
        password,
        user["password_hash"],
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )


    request.session["user_id"] = user["id"]


    return {
        "success": True,
        "message": "Login successful.",
        "redirect": "/dashboard",
    }


# ============================================================
# LOGOUT
# ============================================================

@app.get("/logout")
async def logout(
    request: Request,
):

    request.session.clear()

    return RedirectResponse(
        "/",
        status_code=303,
    )


# ============================================================
# CURRENT USER
# ============================================================

@app.get("/api/me")
async def current_user(
    request: Request,
):

    user = get_current_user(
        request
    )


    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Not authenticated.",
        )


    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "Visual Defect Detection API",
        "device": str(DEVICE),
        "model_exists": MODEL_PATH.exists(),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
async def model_info():

    return {
        "model": "EfficientNet-B0",
        "task": "Binary Visual Defect Classification",
        "classes": [
            "Normal",
            "Defective",
        ],
        "device": str(DEVICE),
        "model_exists": MODEL_PATH.exists(),
        "input_size": "256x640",
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
async def api_predict(
    request: Request,
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    user = get_current_user(request)

    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Please login before using the prediction service.",
        )


    # --------------------------------------------------------
    # FILE TYPE
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }


    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported.",
        )


    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise HTTPException(
            status_code=500,
            detail="Model file not found.",
        )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:

        contents = await file.read()


        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )


        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


    except HTTPException:
        raise


    except Exception:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        result = predict_image(
            image
        )


        result["filename"] = (
            file.filename
        )


        result["user"] = (
            user["email"]
        )


        return JSONResponse(
            content=result
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
):

    return await api_predict(
        request,
        file,
    )