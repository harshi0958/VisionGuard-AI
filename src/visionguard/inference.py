from __future__ import annotations

import base64
import csv
import io
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "artifacts" / "patchcore_memory.pth"
THRESHOLDS_PATH = ROOT / "reports" / "patchcore_v2" / "patchcore_thresholds.csv"

IMAGE_SIZE = 224
FEATURE_SIZE = 14
QUERY_CHUNK_SIZE = 32
DEVICE = torch.device("cpu")


def _read_thresholds() -> dict[str, float]:
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Threshold report not found: {THRESHOLDS_PATH}"
        )

    with THRESHOLDS_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        values = {row["metric"]: float(row["value"]) for row in rows}

    required = (
        "image_threshold_validation_p95",
        "pixel_threshold_validation_p99_5",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing threshold values: {missing}")

    return {
        "image": values["image_threshold_validation_p95"],
        "pixel": values["pixel_threshold_validation_p99_5"],
    }


@lru_cache(maxsize=1)
def _load_model() -> dict:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"PatchCore memory checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("backbone") != "resnet18":
        raise ValueError("Unsupported backbone in checkpoint.")

    memory_bank = checkpoint.get("memory_bank")
    if not isinstance(memory_bank, torch.Tensor):
        raise ValueError("Checkpoint does not contain a valid memory_bank.")

    memory_bank = memory_bank.to(device=DEVICE, dtype=torch.float32).contiguous()

    if memory_bank.ndim != 2 or memory_bank.shape[1] != 256:
        raise ValueError(
            f"Unexpected memory bank shape: {tuple(memory_bank.shape)}"
        )

    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)

    feature_extractor = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    ).to(DEVICE).eval()

    mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(
        1, 3, 1, 1
    )

    thresholds = _read_thresholds()

    return {
        "feature_extractor": feature_extractor,
        "memory_bank": memory_bank,
        "mean": mean,
        "std": std,
        "thresholds": thresholds,
        "metadata": {
            "model": "PatchCore-style anomaly detection",
            "backbone": "ResNet18",
            "image_size": int(checkpoint["image_size"]),
            "feature_size": int(checkpoint["feature_size"]),
            "feature_channels": int(checkpoint["feature_channels"]),
            "memory_patch_count": int(checkpoint["memory_patch_count"]),
            "training_images": int(checkpoint["training_images"]),
            "device": "CPU",
            "category": "bottle",
        },
    }


def _encode_png(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode prediction image.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_model_info() -> dict:
    model = _load_model()
    return {
        **model["metadata"],
        "image_threshold": model["thresholds"]["image"],
        "pixel_threshold": model["thresholds"]["pixel"],
        "status": "ready",
    }


def predict_image(image: Image.Image) -> dict:
    """
    Analyze one image using the trained memory bank.
    The model is intended for the MVTec AD bottle category.
    """
    model = _load_model()

    image = ImageOps.exif_transpose(image).convert("RGB")
    original = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(original, dtype=np.uint8).copy()
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = image_tensor.to(dtype=torch.float32).div_(255.0)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = (image_tensor - model["mean"]) / model["std"]

    with torch.inference_mode():
        features = model["feature_extractor"](image_tensor)
        features = F.adaptive_avg_pool2d(
            features, (FEATURE_SIZE, FEATURE_SIZE)
        )

        # One normalized descriptor per spatial feature location.
        descriptors = features.squeeze(0).permute(1, 2, 0).reshape(-1, 256)
        descriptors = F.normalize(descriptors, p=2, dim=1)

        memory_bank = model["memory_bank"]
        distances = []

        # Limit temporary distance-matrix size for CPU inference.
        for start in range(0, descriptors.shape[0], QUERY_CHUNK_SIZE):
            query_chunk = descriptors[start:start + QUERY_CHUNK_SIZE]
            chunk_distances = torch.cdist(query_chunk, memory_bank)
            distances.append(chunk_distances.min(dim=1).values)

        patch_distances = torch.cat(distances)
        anomaly_map = patch_distances.reshape(
            FEATURE_SIZE, FEATURE_SIZE
        ).unsqueeze(0).unsqueeze(0)

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    anomaly_map = anomaly_map.squeeze().cpu().numpy().astype(np.float32)
    image_score = float(anomaly_map.mean())

    image_threshold = model["thresholds"]["image"]
    pixel_threshold = model["thresholds"]["pixel"]

    prediction = "anomaly" if image_score > image_threshold else "normal"
    mask = (anomaly_map > pixel_threshold).astype(np.uint8) * 255

    # Scale the heatmap consistently relative to the pixel threshold.
    display_map = np.clip(
        anomaly_map / max(pixel_threshold * 2.0, 1e-8), 0.0, 1.0
    )
    display_map = (display_map * 255).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(display_map, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.55, colored_heatmap, 0.45, 0)

    return {
        "prediction": prediction,
        "is_anomaly": prediction == "anomaly",
        "image_score": image_score,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "anomaly_map": _encode_png(display_map),
        "heatmap": _encode_png(
            cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        ),
        "overlay": _encode_png(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ),
        "mask": _encode_png(mask),
        "image_size": {"width": IMAGE_SIZE, "height": IMAGE_SIZE},
        "category": "bottle",
        "model": model["metadata"]["model"],
        "disclaimer": (
            "Experimental bottle-category anomaly detection. "
            "Predictions are not a guarantee of defect-free products."
        ),
    }
