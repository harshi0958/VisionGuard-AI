
from pathlib import Path
import csv
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw" / "bottle"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle_split.json"
MODEL_PATH = ROOT / "artifacts" / "baseline_autoencoder.pth"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

IMAGE_SIZE = 128
CATEGORIES = ["broken_large", "broken_small", "contamination"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_PERCENTILE = 95
PIXEL_PERCENTILE = 99.5


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def image_paths(folder):
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    return tensor.unsqueeze(0), array


def score_image(model, path):
    tensor, original = load_image(path)

    with torch.inference_mode():
        reconstructed = (
            model(tensor)[0].permute(1, 2, 0).cpu().numpy()
        )

    error_map = np.mean(
        (original - reconstructed) ** 2,
        axis=2,
    )

    return float(error_map.mean()), original, reconstructed, error_map


def load_mask(path):
    with Image.open(path) as image:
        image = image.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(image)

    return mask > 0


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(category, path, score, original, reconstructed, error_map):
    normalized = error_map / (error_map.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Input")

    axes[1].imshow(np.clip(reconstructed, 0, 1))
    axes[1].set_title("Reconstruction")

    axes[2].imshow(original)
    axes[2].imshow(
        normalized,
        cmap="jet",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Reconstruction error")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{category}/{path.name} | score={score:.6f}")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{category}_{path.stem}_evaluation.png",
        dpi=130,
    )
    plt.close(fig)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {MODEL_PATH}")

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Dataset split missing: {SPLIT_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    validation_paths = [
        DATA_ROOT / "train" / "good" / name
        for name in split["validation"]
    ]

    missing_validation = [p for p in validation_paths if not p.is_file()]
    if missing_validation:
        raise FileNotFoundError(
            f"Missing validation image: {missing_validation[0]}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 1. Calibrate thresholds using validation images ONLY.
    validation_scores = []
    validation_error_maps = []

    for path in validation_paths:
        score, _, _, error_map = score_image(model, path)
        validation_scores.append(score)
        validation_error_maps.append(error_map)

    image_threshold = float(
        np.percentile(validation_scores, IMAGE_PERCENTILE)
    )

    pixel_threshold = float(
        np.percentile(
            np.concatenate([
                error_map.ravel()
                for error_map in validation_error_maps
            ]),
            PIXEL_PERCENTILE,
        )
    )

    # 2. Evaluate on the untouched test set.
    normal_paths = image_paths(DATA_ROOT / "test" / "good")

    if not normal_paths:
        raise RuntimeError("No independent normal test images found.")

    records = []
    all_labels = []
    all_scores = []
    defect_maps = []
    defect_masks = []
    missing_masks = []
    matched_masks = 0

    # Normal test images have label 0.
    for path in normal_paths:
        score, _, _, _ = score_image(model, path)

        all_labels.append(0)
        all_scores.append(score)

        records.append({
            "category": "good",
            "filename": path.name,
            "label": 0,
            "mean_pixel_error": f"{score:.8f}",
            "image_threshold": f"{image_threshold:.8f}",
            "prediction": int(score > image_threshold),
            "mask_available": "",
        })

    # Defect test images have label 1.
    for category in CATEGORIES:
        paths = image_paths(DATA_ROOT / "test" / category)

        for path in paths:
            score, original, reconstructed, error_map = score_image(
                model, path
            )

            all_labels.append(1)
            all_scores.append(score)

            mask_path = (
                DATA_ROOT / "ground_truth" / category
                / f"{path.stem}_mask.png"
            )

            mask_available = mask_path.is_file()

            if mask_available:
                defect_maps.append(error_map)
                defect_masks.append(load_mask(mask_path))
                matched_masks += 1
            else:
                missing_masks.append(f"{category}/{mask_path.name}")

            records.append({
                "category": category,
                "filename": path.name,
                "label": 1,
                "mean_pixel_error": f"{score:.8f}",
                "image_threshold": f"{image_threshold:.8f}",
                "prediction": int(score > image_threshold),
                "mask_available": mask_available,
            })

            save_heatmap(
                category,
                path,
                score,
                original,
                reconstructed,
                error_map,
            )

    # Image-level metrics.
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
        labels,
        predictions,
        labels=[0, 1],
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

    # Pixel-level metrics on defect test images with masks.
    pixel_rows = [
        {"metric": "pixel_threshold_validation_p99_5", "value": pixel_threshold},
        {"metric": "defect_images_with_masks", "value": matched_masks},
    ]

    if defect_maps:
        pixel_scores = np.concatenate([
            error_map.ravel() for error_map in defect_maps
        ])

        pixel_labels = np.concatenate([
            mask.astype(np.uint8).ravel() for mask in defect_masks
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

        dice_denominator = predicted_positive + actual_positive
        dice = (
            2 * intersection / dice_denominator
            if dice_denominator else float("nan")
        )

        pixel_rows.extend([
            {"metric": "pixel_ROC_AUC", "value": pixel_auc},
            {"metric": "pixel_IoU", "value": iou},
            {"metric": "pixel_Dice", "value": dice},
            {
                "metric": "pixel_threshold_note",
                "value": "Calibrated on normal validation images; preliminary",
            },
        ])
    else:
        pixel_rows.append({
            "metric": "pixel_metrics_status",
            "value": "No matching defect masks found",
        })

    calibration_rows = [
        {
            "metric": "validation_image_threshold",
            "value": image_threshold,
        },
        {
            "metric": "validation_pixel_threshold",
            "value": pixel_threshold,
        },
        {
            "metric": "validation_image_count",
            "value": len(validation_paths),
        },
        {
            "metric": "image_percentile",
            "value": IMAGE_PERCENTILE,
        },
        {
            "metric": "pixel_percentile",
            "value": PIXEL_PERCENTILE,
        },
    ]

    save_csv(
        REPORT_DIR / "visionguard_image_metrics.csv",
        image_rows,
        ["metric", "value"],
    )

    save_csv(
        REPORT_DIR / "visionguard_pixel_metrics.csv",
        pixel_rows,
        ["metric", "value"],
    )

    save_csv(
        REPORT_DIR / "visionguard_test_predictions.csv",
        records,
        [
            "category", "filename", "label", "mean_pixel_error",
            "image_threshold", "prediction", "mask_available",
        ],
    )

    save_csv(
        REPORT_DIR / "visionguard_thresholds.csv",
        calibration_rows,
        ["metric", "value"],
    )

    print("\n=== VisionGuard AI: Validation-Calibrated Evaluation ===")
    print(f"Training images: {checkpoint.get('training_images', 'unknown')}")
    print(f"Validation images: {len(validation_paths)}")
    print(f"Normal test images: {len(normal_paths)}")
    print(f"Defect test images: {int(labels.sum())}")
    print(f"Matched ground-truth masks: {matched_masks}")

    print(f"\nImage threshold (validation p{IMAGE_PERCENTILE}): {image_threshold:.6f}")
    print(f"Image ROC-AUC: {image_auc:.4f}")
    print(f"Image Average Precision: {image_ap:.4f}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print(f"\nPixel threshold (validation p{PIXEL_PERCENTILE}): {pixel_threshold:.6f}")

    for row in pixel_rows:
        if row["metric"] in {"pixel_ROC_AUC", "pixel_IoU", "pixel_Dice"}:
            print(f"{row['metric']}: {float(row['value']):.4f}")

    if missing_masks:
        print("\nMissing masks:")
        for name in missing_masks:
            print(f"  - {name}")

    print("\nReports saved:")
    print(REPORT_DIR / "visionguard_image_metrics.csv")
    print(REPORT_DIR / "visionguard_pixel_metrics.csv")
    print(REPORT_DIR / "visionguard_test_predictions.csv")
    print(REPORT_DIR / "visionguard_thresholds.csv")
    print(f"Heatmaps: {FIGURE_DIR}")
    print("\nNote: small dataset; metrics are preliminary.")


if __name__ == "__main__":
    main()
