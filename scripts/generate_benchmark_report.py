from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PATCHCORE_DIR = REPORTS / "patchcore_v2"
OUTPUT_DIR = REPORTS / "benchmark"
FIGURES_DIR = OUTPUT_DIR / "figures"


def read_metrics(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics file not found: {path}")

    metrics = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            key = row.get("metric")
            value = row.get("value")
            if not key or value is None:
                continue
            try:
                metrics[key.strip()] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def find_metric(metrics, *names):
    for name in names:
        if name in metrics and math.isfinite(metrics[name]):
            return metrics[name]
    return None


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_image = read_metrics(
        REPORTS / "visionguard_image_metrics.csv"
    )
    baseline_pixel = read_metrics(
        REPORTS / "visionguard_pixel_metrics.csv"
    )
    patch_image = read_metrics(
        PATCHCORE_DIR / "patchcore_image_metrics.csv"
    )
    patch_pixel = read_metrics(
        PATCHCORE_DIR / "patchcore_pixel_metrics.csv"
    )

    models = {
        "Autoencoder baseline": {
            "Image ROC-AUC": find_metric(
                baseline_image, "image_ROC_AUC", "roc_auc"
            ),
            "Image Average Precision": find_metric(
                baseline_image, "image_average_precision",
                "image_AP", "average_precision"
            ),
            "Pixel ROC-AUC": find_metric(
                baseline_pixel, "pixel_ROC_AUC"
            ),
            "Pixel IoU": find_metric(
                baseline_pixel, "pixel_IoU", "pixel_iou"
            ),
            "Pixel Dice": find_metric(
                baseline_pixel, "pixel_Dice", "pixel_dice"
            ),
        },
        "PatchCore-style v2": {
            "Image ROC-AUC": find_metric(
                patch_image, "image_ROC_AUC", "roc_auc"
            ),
            "Image Average Precision": find_metric(
                patch_image, "image_average_precision",
                "image_AP", "average_precision"
            ),
            "Pixel ROC-AUC": find_metric(
                patch_pixel, "pixel_ROC_AUC"
            ),
            "Pixel IoU": find_metric(
                patch_pixel, "pixel_IoU", "pixel_iou"
            ),
            "Pixel Dice": find_metric(
                patch_pixel, "pixel_Dice", "pixel_dice"
            ),
        },
    }

    rows = []
    for metric_name in next(iter(models.values())):
        row = {"metric": metric_name}
        for model_name, values in models.items():
            row[model_name] = values[metric_name]
        rows.append(row)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    with comparison_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "Autoencoder baseline",
                "PatchCore-style v2",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    "" if value is None else value
                ) if key != "metric" else value
                for key, value in row.items()
            })

    # Comparison chart: each metric remains on its original 0-1 scale.
    chart_metrics = [
        "Image ROC-AUC",
        "Image Average Precision",
        "Pixel ROC-AUC",
        "Pixel IoU",
        "Pixel Dice",
    ]
    baseline_values = [
        models["Autoencoder baseline"][name]
        if models["Autoencoder baseline"][name] is not None else 0
        for name in chart_metrics
    ]
    patch_values = [
        models["PatchCore-style v2"][name]
        if models["PatchCore-style v2"][name] is not None else 0
        for name in chart_metrics
    ]

    positions = list(range(len(chart_metrics)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(
        [p - width / 2 for p in positions],
        baseline_values,
        width,
        label="Autoencoder baseline",
    )
    bars2 = ax.bar(
        [p + width / 2 for p in positions],
        patch_values,
        width,
        label="PatchCore-style v2",
    )

    ax.set_title("VisionGuard AI — Model Benchmark")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(positions)
    ax.set_xticklabels(chart_metrics, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars, model_name in [
        (bars1, "baseline"),
        (bars2, "patchcore"),
    ]:
        values = (
            baseline_values if model_name == "baseline"
            else patch_values
        )
        for bar, metric_name, value in zip(
            bars, chart_metrics, values
        ):
            actual = models[
                "Autoencoder baseline"
                if model_name == "baseline"
                else "PatchCore-style v2"
            ][metric_name]
            if actual is not None:
                ax.annotate(
                    f"{actual:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    fig.tight_layout()
    chart_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    image_threshold = find_metric(
        patch_image, "image_threshold_validation_p95"
    )
    pixel_threshold = find_metric(
        patch_pixel, "pixel_threshold_validation_p99_5"
    )
    tn = find_metric(patch_image, "true_negative")
    fp = find_metric(patch_image, "false_positive")
    fn = find_metric(patch_image, "false_negative")
    tp = find_metric(patch_image, "true_positive")
    normal_count = find_metric(patch_image, "normal_test_images")
    defect_count = find_metric(patch_image, "defect_test_images")
    mask_count = find_metric(
        patch_pixel, "defect_images_with_masks"
    )

    report_lines = [
        "# VisionGuard AI — Model Benchmark Report",
        "",
        "## 1. Objective",
        "",
        "Compare a reconstruction-based convolutional autoencoder "
        "with a pretrained ResNet18 nearest-neighbour, PatchCore-style "
        "anomaly detection approach.",
        "",
        "## 2. Dataset and evaluation",
        "",
        "- Category: MVTec AD bottle.",
        "- Training split: 48 normal images.",
        "- Validation split: 12 normal images.",
        f"- Normal test images: {int(normal_count) if normal_count is not None else 'N/A'}.",
        f"- Defect test images: {int(defect_count) if defect_count is not None else 'N/A'}.",
        f"- Defect images with matched masks for localization: {int(mask_count) if mask_count is not None else 'N/A'}.",
        "- The PatchCore-style memory bank was built from training images only.",
        "- Image threshold: 95th percentile of validation image scores.",
        "- Pixel threshold: 99.5th percentile of validation normal-image anomaly-map values.",
        "- The held-out test set is used for final reported metrics, not threshold selection.",
        "",
        "## 3. Model comparison",
        "",
        "| Metric | Autoencoder baseline | PatchCore-style v2 |",
        "|---|---:|---:|",
    ]

    for row in rows:
        report_lines.append(
            f"| {row['metric']} | "
            f"{fmt(row['Autoencoder baseline'])} | "
            f"{fmt(row['PatchCore-style v2'])} |"
        )

    report_lines += [
        "",
        "![Model comparison chart](figures/model_comparison.png)",
        "",
        "## 4. PatchCore-style classification at the calibrated threshold",
        "",
        f"- Image threshold: {fmt(image_threshold)}.",
        f"- Pixel threshold: {fmt(pixel_threshold)}.",
        f"- True negatives: {int(tn) if tn is not None else 'N/A'}.",
        f"- False positives: {int(fp) if fp is not None else 'N/A'}.",
        f"- False negatives: {int(fn) if fn is not None else 'N/A'}.",
        f"- True positives: {int(tp) if tp is not None else 'N/A'}.",
        "",
        "## 5. Methodology notes",
        "",
        "- The autoencoder reconstructs an input image and uses reconstruction error as an anomaly signal.",
        "- The PatchCore-style approach extracts pretrained ResNet18 layer-3 features, normalizes patch descriptors, and measures nearest-neighbour distances to a normal-training memory bank.",
        "- The implementation is PatchCore-style, not a reproduction of every component of the original PatchCore algorithm.",
        "- Pixel localization metrics are computed over defect test images with matched ground-truth masks. They should not be interpreted as a full normal-plus-defect pixel benchmark.",
        "- Pixel IoU and Dice use a threshold calibrated from validation normal images.",
        "",
        "## 6. Limitations",
        "",
        "- The test set is small; the results have substantial sampling uncertainty.",
        "- Only the bottle category is evaluated in this experiment.",
        "- False positives remain possible at the current image threshold.",
        "- Heatmaps are approximate localization maps and may not precisely trace defect boundaries.",
        "- Results do not establish production readiness or generalization to other products, cameras, lighting conditions, or manufacturing environments.",
        "- The dataset's license must be reviewed before any commercial use.",
        "",
        "## 7. Conclusion",
        "",
        "Use the measured comparison above to select the current research baseline. "
        "The PatchCore-style model is a promising candidate on this experiment, "
        "but further validation on more images and additional categories is needed "
        "before making broad accuracy claims.",
        "",
        "## 8. Generated files",
        "",
        "- `model_comparison.csv` — comparison metrics.",
        "- `figures/model_comparison.png` — comparison chart.",
        "- The original baseline and PatchCore-style evaluation files remain unchanged.",
        "",
    ]

    report_path = OUTPUT_DIR / "benchmark_report.md"
    report_path.write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("=== VisionGuard AI Benchmark Report Created ===")
    print(f"Report: {report_path}")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Chart: {chart_path}")
    print("\nMetric comparison:")
    for row in rows:
        print(
            f"{row['metric']}: "
            f"baseline={fmt(row['Autoencoder baseline'])}, "
            f"patchcore={fmt(row['PatchCore-style v2'])}"
        )


if __name__ == "__main__":
    main()
