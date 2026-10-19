from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "bottle" / "train" / "good"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle_split.json"
ARTIFACT_DIR = ROOT / "artifacts"

OUTPUT_PATH = ARTIFACT_DIR / "patchcore_memory.pth"
IMAGE_SIZE = 224
FEATURE_SIZE = 14
PATCH_BUDGET = 5000
SEED = 42


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

    return ((tensor - mean) / std).unsqueeze(0)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Missing split file: {SPLIT_PATH}")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_names = split["train"]

    if len(train_names) != 48:
        print(f"Note: split contains {len(train_names)} training images.")

    missing = [name for name in train_names if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training image: {missing[0]}")

    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)

    # Combine intermediate features from ResNet layer 2 and layer 3.
    feature_extractor = torch.nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )
    feature_extractor.eval()

    feature_batches = []

    print("=== VisionGuard AI v2: Building Memory Bank ===")
    print(f"Training images: {len(train_names)}")
    print("Backbone: pretrained ResNet18")
    print("Device: CPU")
    print("Extracting normal-image features...")

    with torch.inference_mode():
        for index, name in enumerate(train_names, start=1):
            image_tensor = load_tensor(DATA_DIR / name, weights)
            features = feature_extractor(image_tensor)

            # Standardize spatial feature-map dimensions.
            features = F.adaptive_avg_pool2d(
                features, (FEATURE_SIZE, FEATURE_SIZE)
            )

            # [C,H,W] -> [H*W,C]
            patches = features[0].permute(1, 2, 0).reshape(
                -1, features.shape[1]
            )

            # L2-normalize patch descriptors.
            patches = F.normalize(patches, p=2, dim=1)
            feature_batches.append(patches.cpu())

            if index % 8 == 0 or index == len(train_names):
                print(f"Processed {index}/{len(train_names)} images")

    memory = torch.cat(feature_batches, dim=0)

    # Random subset keeps CPU nearest-neighbor search manageable.
    generator = torch.Generator().manual_seed(SEED)
    selected_count = min(PATCH_BUDGET, len(memory))
    indices = torch.randperm(
        len(memory), generator=generator
    )[:selected_count]

    memory = memory[indices].contiguous()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "memory_bank": memory,
            "backbone": "resnet18",
            "weights": "ResNet18_Weights.DEFAULT",
            "image_size": IMAGE_SIZE,
            "feature_size": FEATURE_SIZE,
            "feature_channels": int(memory.shape[1]),
            "memory_patch_count": int(memory.shape[0]),
            "training_images": len(train_names),
            "seed": SEED,
        },
        OUTPUT_PATH,
    )

    print("\nMemory bank created successfully.")
    print(f"Memory shape: {tuple(memory.shape)}")
    print(f"Memory patches: {len(memory)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("Validation and test images were not used to build the memory bank.")


if __name__ == "__main__":
    main()
