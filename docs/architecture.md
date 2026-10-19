# VisionGuard AI — Architecture

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
