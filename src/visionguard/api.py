from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from visionguard.inference import get_model_info, predict_image


APP_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

app = FastAPI(
    title="VisionGuard AI API",
    description=(
        "Experimental bottle-category anomaly detection API "
        "using a trained PatchCore-style model."
    ),
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "VisionGuard AI API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "success": True,
        "status": "running",
        "service": "VisionGuard AI API",
        "version": APP_VERSION,
    }


@app.get("/api/model-info")
def model_info() -> dict:
    try:
        info = get_model_info()
        return {"success": True, "model": info}
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {exc}",
        ) from exc


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Upload PNG, JPEG, WEBP, or BMP.",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as probe:
            probe.verify()

        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are invalid.",
                )
            result = predict_image(image.copy())

    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid supported image.",
        ) from exc
    except Exception as exc:
        # Keep unexpected internal details out of the public response.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend terminal logs.",
        ) from exc

    return {
        "success": True,
        "filename": Path(file.filename or "upload").name,
        "result": result,
    }
