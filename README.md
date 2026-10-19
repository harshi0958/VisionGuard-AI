# VisionGuard AI
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
| Image ROC-AUC | 0.5649 | 0.9675 |
| Image Average Precision | 0.7325 | 0.9812 |
| Pixel ROC-AUC | 0.6912 | 0.9674 |
| Pixel IoU | 0.0514 | 0.3479 |
| Pixel Dice | 0.0977 | 0.5162 |

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
