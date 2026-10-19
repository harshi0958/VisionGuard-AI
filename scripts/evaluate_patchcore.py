from pathlib import Path
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw" / "bottle"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle_split.json"
MEMORY_PATH = ROOT / "artifacts" / "patchcore_memory.pth"
OUTPUT_DIR = ROOT / "reports" / "patchcore_v2"
FIGURE_DIR = OUTPUT_DIR / "figures"

IMAGE_SIZE = 224
FEATURE_SIZE = 14
CATEGORIES = ["broken_large", "broken_small", "contamination"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_PERCENTILE = 95
PIXEL_PERCENTILE = 99.5

DEVICE = torch.device("cpu")


def image_paths(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def load_tensor(path, weights):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(3, 1, 1)

    return ((tensor - mean) / std).unsqueeze(0), array


def extract_features(model, tensor):
    with torch.inference_mode():
        features = model(tensor.to(DEVICE))
        features = F.adaptive_avg_pool2d(
            features, (FEATURE_SIZE, FEATURE_SIZE)
        )

        patches = features[0].permute(1, 2, 0).reshape(
            -1, features.shape[1]
        )
        patches = F.normalize(patches, p=2, dim=1)

    return patches


def score_image(model, memory, path, weights):
    tensor, original = load_tensor(path, weights)
    patches = extract_features(model, tensor)

    # Chunk queries to keep CPU memory use manageable.
    min_distances = []
    memory = memory.to(DEVICE)

    with torch.inference_mode():
        for start in range(0, len(patches), 32):
            query = patches[start:start + 32]

            # Euclidean distance for L2-normalized feature vectors.
            distances = torch.cdist(query, memory, p=2)
            nearest = distances.min(dim=1).values
            min_distances.append(nearest)

    patch_scores = torch.cat(min_distances)
    anomaly_map = patch_scores.reshape(
        1, 1, FEATURE_SIZE, FEATURE_SIZE
    )

    # Upsample the patch anomaly map to image resolution.
    anomaly_map = F.interpolate(
        anomaly_map,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    return float(anomaly_map.mean()), original, anomaly_map


def load_mask(path):
    with Image.open(path) as image:
        image = image.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )
        return np.asarray(image) > 0


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_visual(category, path, original, anomaly_map, image_score):
    normalized = anomaly_map / (anomaly_map.max() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(original)
    axes[0].set_title("Input")

    axes[1].imshow(original)
    axes[1].imshow(
        normalized, cmap="jet", alpha=0.55, vmin=0, vmax=1
    )
    axes[1].set_title("Patch anomaly map")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{category}/{path.name} | anomaly score={image_score:.5f}"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{category}_{path.stem}_patchcore.png",
        dpi=130,
    )
    plt.close(fig)


def main():
    if not MEMORY_PATH.is_file():
        raise FileNotFoundError(f"Memory bank missing: {MEMORY_PATH}")
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Dataset split missing: {SPLIT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    validation_paths = [
        DATA_ROOT / "train" / "good" / name
        for name in split["validation"]
    ]

    missing = [p for p in validation_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation image: {missing[0]}")

    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)

    feature_extractor = torch.nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    ).to(DEVICE)
    feature_extractor.eval()

    checkpoint = torch.load(
        MEMORY_PATH,
        map_location="cpu",
        weights_only=True,
    )
    memory = checkpoint["memory_bank"].float()

    print("=== VisionGuard AI v2: PatchCore-style Evaluation ===")
    print(f"Memory patches: {len(memory)}")
    print(f"Validation images: {len(validation_paths)}")
    print("Calibrating thresholds on validation images...")

    validation_scores = []
    validation_maps = []

    for index, path in enumerate(validation_paths, start=1):
        score, _, anomaly_map = score_image(
            feature_extractor, memory, path, weights
        )
        validation_scores.append(score)
        validation_maps.append(anomaly_map)

        print(f"Validation {index}/{len(validation_paths)}: {path.name}")

    image_threshold = float(
        np.percentile(validation_scores, IMAGE_PERCENTILE)
    )
    pixel_threshold = float(
        np.percentile(
            np.concatenate([m.ravel() for m in validation_maps]),
            PIXEL_PERCENTILE,
        )
    )

    all_labels = []
    all_scores = []
    records = []
    defect_maps = []
    defect_masks = []
    missing_masks = []
    matched_masks = 0

    normal_paths = image_paths(DATA_ROOT / "test" / "good")
    if not normal_paths:
        raise RuntimeError("No normal test images found.")

    print("\nEvaluating normal test images...")
    for path in normal_paths:
        score, _, anomaly_map = score_image(
            feature_extractor, memory, path, weights
        )

        all_labels.append(0)
        all_scores.append(score)
        records.append({
            "category": "good",
            "filename": path.name,
            "label": 0,
            "anomaly_score": f"{score:.8f}",
            "image_threshold": f"{image_threshold:.8f}",
            "prediction": int(score > image_threshold),
            "mask_available": "",
        })

    print("Evaluating defect test images...")
    for category in CATEGORIES:
        for path in image_paths(DATA_ROOT / "test" / category):
            score, original, anomaly_map = score_image(
                feature_extractor, memory, path, weights
            )

            all_labels.append(1)
            all_scores.append(score)

            mask_path = (
                DATA_ROOT / "ground_truth" / category
                / f"{path.stem}_mask.png"
            )
            has_mask = mask_path.is_file()

            if has_mask:
                defect_maps.append(anomaly_map)
                defect_masks.append(load_mask(mask_path))
                matched_masks += 1
            else:
                missing_masks.append(f"{category}/{mask_path.name}")

            records.append({
                "category": category,
                "filename": path.name,
                "label": 1,
                "anomaly_score": f"{score:.8f}",
                "image_threshold": f"{image_threshold:.8f}",
                "prediction": int(score > image_threshold),
                "mask_available": has_mask,
            })

            save_visual(
                category, path, original, anomaly_map, score
            )

    labels = np.asarray(all_labels, dtype=np.int32)
    scores = np.asarray(all_scores, dtype=np.float64)
    predictions = (scores > image_threshold).astype(np.int32)

    image_auc = safe_auc(labels, scores)
    image_ap = (
        float(average_precision_score(labels, scores))
        if len(np.unique(labels)) == 2
        else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    image_rows = [
        {"metric": "image_ROC_AUC", "value": image_auc},
        {"metric": "image_average_precision", "value": image_ap},
        {"metric": "image_threshold_validation_p95", "value": image_threshold},
        {"metric": "true_negative", "value": int(tn)},
        {"metric": "false_positive", "value": int(fp)},
        {"metric": "false_negative", "value": int(fn)},
        {"metric": "true_positive", "value": int(tp)},
        {"metric": "validation_images", "value": len(validation_paths)},
        {"metric": "normal_test_images", "value": len(normal_paths)},
        {"metric": "defect_test_images", "value": int(labels.sum())},
    ]

    pixel_rows = [
        {"metric": "pixel_threshold_validation_p99_5", "value": pixel_threshold},
        {"metric": "defect_images_with_masks", "value": matched_masks},
    ]

    if defect_maps:
        pixel_scores = np.concatenate([m.ravel() for m in defect_maps])
        pixel_labels = np.concatenate([
            m.astype(np.uint8).ravel() for m in defect_masks
        ])

        pixel_auc = safe_auc(pixel_labels, pixel_scores)
        pixel_predictions = pixel_scores > pixel_threshold

        intersection = int(np.logical_and(
            pixel_predictions, pixel_labels == 1
        ).sum())
        predicted_positive = int(pixel_predictions.sum())
        actual_positive = int((pixel_labels == 1).sum())
        union = int(np.logical_or(
            pixel_predictions, pixel_labels == 1
        ).sum())

        iou = intersection / union if union else float("nan")
        denom = predicted_positive + actual_positive
        dice = 2 * intersection / denom if denom else float("nan")

        pixel_rows.extend([
            {"metric": "pixel_ROC_AUC", "value": pixel_auc},
            {"metric": "pixel_IoU", "value": iou},
            {"metric": "pixel_Dice", "value": dice},
        ])

    calibration_rows = [
        {"metric": "image_threshold_validation_p95", "value": image_threshold},
        {"metric": "pixel_threshold_validation_p99_5", "value": pixel_threshold},
        {"metric": "validation_images", "value": len(validation_paths)},
        {"metric": "memory_patches", "value": len(memory)},
    ]

    save_csv(
        OUTPUT_DIR / "patchcore_image_metrics.csv",
        image_rows,
        ["metric", "value"],
    )
    save_csv(
        OUTPUT_DIR / "patchcore_pixel_metrics.csv",
        pixel_rows,
        ["metric", "value"],
    )
    save_csv(
        OUTPUT_DIR / "patchcore_test_predictions.csv",
        records,
        [
            "category", "filename", "label", "anomaly_score",
            "image_threshold", "prediction", "mask_available",
        ],
    )
    save_csv(
        OUTPUT_DIR / "patchcore_thresholds.csv",
        calibration_rows,
        ["metric", "value"],
    )

    print("\n=== PatchCore-style Results ===")
    print(f"Image threshold (validation p95): {image_threshold:.6f}")
    print(f"Image ROC-AUC: {image_auc:.4f}")
    print(f"Image Average Precision: {image_ap:.4f}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Pixel threshold (validation p99.5): {pixel_threshold:.6f}")

    for row in pixel_rows:
        if row["metric"] in {"pixel_ROC_AUC", "pixel_IoU", "pixel_Dice"}:
            print(f"{row['metric']}: {float(row['value']):.4f}")

    if missing_masks:
        print("Missing masks:", missing_masks)

    print(f"\nReports saved to: {OUTPUT_DIR}")
    print(f"Heatmaps saved to: {FIGURE_DIR}")
    print("Results are preliminary; compare with the autoencoder baseline.")


if __name__ == "__main__":
    main()
