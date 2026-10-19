from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
BENCHMARK = REPORTS / "benchmark"
DOCS = ROOT / "docs"


def read_csv_metrics(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("metric")
            value = row.get("value")
            if key and value is not None:
                try:
                    result[key.strip()] = float(value)
                except ValueError:
                    pass
    return result


baseline_image = read_csv_metrics(
    REPORTS / "visionguard_image_metrics.csv"
)
baseline_pixel = read_csv_metrics(
    REPORTS / "visionguard_pixel_metrics.csv"
)
patch_image = read_csv_metrics(
    REPORTS / "patchcore_v2" / "patchcore_image_metrics.csv"
)
patch_pixel = read_csv_metrics(
    REPORTS / "patchcore_v2" / "patchcore_pixel_metrics.csv"
)

def metric(data, *keys):
    for key in keys:
        if key in data:
            return f"{data[key]:.4f}"
    return "N/A"


readme = r"""# VisionGuard AI
### Visual Anomaly Detection and Defect Localization

VisionGuard AI is a computer-vision research project that explores
unsupervised anomaly detection for industrial visual inspection.
It compares a convolutional autoencoder baseline with a pretrained
ResNet18 nearest-neighbour approach inspired by PatchCore.

The prototype learns visual patterns from normal training images and
produces anomaly scores and heatmaps for potentially defective images.

## Key Features

- Normal-only training for the autoencoder baseline.
- Pretrained ResNet18 feature extraction for the PatchCore-style model.
- Memory-bank nearest-neighbour anomaly scoring.
- Image-level anomaly classification scores.
- Pixel-level anomaly heatmaps and mask-based evaluation.
- Validation-calibrated image and pixel thresholds.
- Reproducible train/validation split with a fixed seed.
- CSV metrics, prediction records, heatmaps, and benchmark charts.

## Model Architecture

### 1. Autoencoder baseline
An encoder compresses an input image into a feature representation.
A decoder reconstructs the image. Reconstruction error is used as
an anomaly signal.

### 2. PatchCore-style v2
A pretrained ResNet18 feature extractor produces spatial descriptors.
Descriptors from normal training images form a compact memory bank.
At inference, patch descriptors are compared with the memory bank
using nearest-neighbour distance. The resulting distance map is
upsampled to produce an anomaly heatmap.

This is a PatchCore-style implementation, not a complete reproduction
of the original PatchCore algorithm.

## Current Experimental Results

The following results are generated from this repository's evaluation
CSV files. Results are preliminary and specific to this experiment.

| Metric | Autoencoder baseline | PatchCore-style v2 |
|---|---:|---:|
| Image ROC-AUC | BASELINE_IMAGE_AUC | PATCH_IMAGE_AUC |
| Image Average Precision | BASELINE_IMAGE_AP | PATCH_IMAGE_AP |
| Pixel ROC-AUC | BASELINE_PIXEL_AUC | PATCH_PIXEL_AUC |
| Pixel IoU | BASELINE_PIXEL_IOU | PATCH_PIXEL_IOU |
| Pixel Dice | BASELINE_PIXEL_DICE | PATCH_PIXEL_DICE |

![Benchmark comparison](reports/benchmark/figures/model_comparison.png)

### Dataset and evaluation notes
- Dataset category: MVTec AD bottle.
- Normal training images: 48.
- Normal validation images: 12.
- Test images: 11 normal and 14 defective.
- Thresholds are calibrated using normal validation images.
- Pixel localization metrics use defective test images with matching
  ground-truth masks; they are not a full normal-plus-defect pixel
  evaluation.
- The small test set limits the confidence and generalizability of
  these metrics.

## Project Structure

```text
VisionGuardAI/
├── artifacts/
│   ├── baseline_autoencoder.pth
│   └── patchcore_memory.pth
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── architecture.md
├── reports/
│   ├── benchmark/
│   ├── patchcore_v2/
│   ├── figures/
│   ├── visionguard_image_metrics.csv
│   └── visionguard_pixel_metrics.csv
├── scripts/
│   ├── train_autoencoder.py
│   ├── evaluate_baseline.py
│   ├── build_patchcore_memory.py
│   ├── evaluate_patchcore.py
│   ├── evaluate_patchcore.py
│   ├── generate_benchmark_report.py
│   └── create_dataset_split.py
└── tests/
```

## Environment

The project was developed and evaluated on CPU using Python 3.13.5.
The environment includes PyTorch, torchvision, OpenCV, NumPy, pandas,
Matplotlib, Pillow, and scikit-learn.

Use the project's existing virtual environment for reproducibility.

### Run evaluation scripts

From PowerShell at the project root:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_baseline.py
.\.venv\Scripts\python.exe .\scripts\build_patchcore_memory.py
.\.venv\Scripts\python.exe .\scripts\evaluate_patchcore.py
.\.venv\Scripts\python.exe .\scripts\generate_benchmark_report.py
```

These commands can regenerate reports or overwrite existing evaluation
outputs. Back up results before rerunning them if you want to preserve
the current benchmark.

## Limitations and Future Work

- Evaluation currently covers one object category and a small test set.
- False positives and missed defects remain possible.
- Heatmaps approximate anomalous regions and may not trace boundaries
  precisely.
- More categories, repeated experiments, threshold analysis, and
  external validation are required before production use.
- CPU inference speed and memory use should be benchmarked before
  deployment.
- The dataset license and pretrained-weight terms must be reviewed
  before redistribution or commercial deployment.

## Dataset and Licensing

This experiment uses the bottle category from MVTec Anomaly Detection
(MVTec AD). Review the dataset's applicable license and terms before
redistributing data or using it commercially. Do not assume that the
dataset is commercially unrestricted.

## Research Integrity

Report measured results with their evaluation scope. Do not describe
these preliminary scores as guaranteed production accuracy, and do not
claim real-time performance without measured latency results.
"""

replacements = {
    "BASELINE_IMAGE_AUC": metric(
        baseline_image, "image_ROC_AUC", "roc_auc"
    ),
    "PATCH_IMAGE_AUC": metric(
        patch_image, "image_ROC_AUC", "roc_auc"
    ),
    "BASELINE_IMAGE_AP": metric(
        baseline_image, "image_average_precision",
        "image_AP", "average_precision"
    ),
    "PATCH_IMAGE_AP": metric(
        patch_image, "image_average_precision",
        "image_AP", "average_precision"
    ),
    "BASELINE_PIXEL_AUC": metric(
        baseline_pixel, "pixel_ROC_AUC"
    ),
    "PATCH_PIXEL_AUC": metric(
        patch_pixel, "pixel_ROC_AUC"
    ),
    "BASELINE_PIXEL_IOU": metric(
        baseline_pixel, "pixel_IoU", "pixel_iou"
    ),
    "PATCH_PIXEL_IOU": metric(
        patch_pixel, "pixel_IoU", "pixel_iou"
    ),
    "BASELINE_PIXEL_DICE": metric(
        baseline_pixel, "pixel_Dice", "pixel_dice"
    ),
    "PATCH_PIXEL_DICE": metric(
        patch_pixel, "pixel_Dice", "pixel_dice"
    ),
}

for key, value in replacements.items():
    readme = readme.replace(key, value)

architecture = """# VisionGuard AI — Architecture

## Overview

VisionGuard AI currently contains two independent anomaly-detection
pipelines: a convolutional autoencoder baseline and a PatchCore-style
nearest-neighbour model using pretrained ResNet18 features.

## Pipeline A: Autoencoder

1. Load normal training images.
2. Train the encoder-decoder reconstruction network.
3. Select the checkpoint using validation reconstruction loss.
4. Calculate reconstruction-error maps for evaluation images.
5. Calibrate image and pixel thresholds on normal validation data.
6. Evaluate image ranking, classification, and pixel localization.

## Pipeline B: PatchCore-style model

1. Load the deterministic normal-only training split.
2. Resize and normalize images for pretrained ResNet18.
3. Extract spatial feature maps through layer 3.
4. L2-normalize the spatial patch descriptors.
5. Sample a compact memory bank from training descriptors.
6. Extract descriptors for validation or evaluation images.
7. Calculate nearest-neighbour distances to the memory bank.
8. Upsample patch distances to image resolution.
9. Calibrate thresholds using normal validation images.
10. Evaluate image-level and mask-based pixel-level metrics.

## Data boundaries

- Training images construct the model and memory bank.
- Normal validation images calibrate thresholds and select the baseline
  checkpoint.
- Test images are reserved for final evaluation.
- Ground-truth defect masks are used to calculate localization metrics,
  not to construct the normal training memory bank.

## Current artifacts

- `artifacts/baseline_autoencoder.pth`
- `artifacts/patchcore_memory.pth`
- `reports/visionguard_image_metrics.csv`
- `reports/visionguard_pixel_metrics.csv`
- `reports/patchcore_v2/patchcore_image_metrics.csv`
- `reports/patchcore_v2/patchcore_pixel_metrics.csv`
- `reports/benchmark/benchmark_report.md`
- `reports/benchmark/model_comparison.csv`
- `reports/benchmark/figures/model_comparison.png`

## Evaluation cautions

The current experiment uses one object category and a small test set.
Pixel metrics are calculated over defective test images with matched
masks, so their scope must be stated in reports. A high ROC-AUC does
not guarantee that a chosen operating threshold will avoid false
alarms. The current model is a research prototype, not a validated
production inspection system.
"""

DOCS.mkdir(parents=True, exist_ok=True)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
(DOCS / "architecture.md").write_text(
    architecture, encoding="utf-8"
)

print("=== Documentation created successfully ===")
print(f"README: {ROOT / 'README.md'}")
print(f"Architecture: {DOCS / 'architecture.md'}")
print("\nBenchmark values inserted from existing CSV files:")
for key, value in replacements.items():
    print(f"{key}: {value}")
