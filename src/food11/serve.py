import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from .data import CLASSES, resize_image

model = None


@asynccontextmanager
async def lifespan(app):
    global model
    if not os.getenv("SKIP_MODEL"):
        import mlflow.pyfunc

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        )
        model = mlflow.pyfunc.load_model("models:/food11@champion")
    try:
        yield
    finally:
        model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile):
    if model is None:
        raise HTTPException(503, "Model unavailable")
    try:
        with Image.open(file.file) as image:
            pixels = np.asarray(resize_image(image), dtype=np.float32) / 255.0
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Invalid image") from exc
    pixels -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    pixels /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    logits = np.asarray(model.predict(batch))[0]
    scores = np.exp(logits - logits.max())
    index = int(scores.argmax())
    return {
        "category": CLASSES[index],
        "confidence": float(scores[index] / scores.sum()),
    }
