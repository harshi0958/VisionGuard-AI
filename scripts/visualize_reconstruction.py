from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "raw" / "bottle" / "train" / "good"
MODEL_PATH = ROOT / "artifacts" / "baseline_autoencoder.pth"
OUTPUT_DIR = ROOT / "reports" / "figures"


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    images = sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

    image_path = images[0]
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(image_path) as image:
        original = image.convert("RGB").resize((128, 128))
        original_array = np.asarray(original, dtype=np.float32) / 255.0

    input_tensor = torch.from_numpy(
        original_array.copy()
    ).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        reconstruction = model(input_tensor)[0]
        reconstructed_array = (
            reconstruction.permute(1, 2, 0).numpy()
        )

    error_map = np.mean(
        (original_array - reconstructed_array) ** 2,
        axis=2,
    )
    mean_error = float(error_map.mean())

    # Scale heatmap for visualization only; not a calibrated defect score.
    display_map = error_map / (error_map.max() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Bottle")

    axes[1].imshow(np.clip(reconstructed_array, 0, 1))
    axes[1].set_title("Autoencoder Reconstruction")

    axes[2].imshow(original_array)
    axes[2].imshow(display_map, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title("Reconstruction Error Heatmap")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"VisionGuard AI | {image_path.name} | "
        f"Mean pixel error: {mean_error:.6f}"
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / "baseline_reconstruction.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("=== VisionGuard AI: Reconstruction Analysis ===")
    print(f"Input image       : {image_path.name}")
    print(f"Mean pixel error  : {mean_error:.6f}")
    print(f"Visualization     : {output_path}")
    print("Analysis completed.")
    print("Note: Heatmap highlights reconstruction differences, not confirmed defects.")


if __name__ == "__main__":
    main()
