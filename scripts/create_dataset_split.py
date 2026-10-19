from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "bottle" / "train" / "good"
OUTPUT = ROOT / "data" / "processed" / "bottle_split.json"

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SEED = 42

files = sorted(
    p.name for p in DATA_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if len(files) < 10:
    raise RuntimeError(f"Not enough images: {len(files)}")

random.Random(SEED).shuffle(files)

split_index = int(len(files) * 0.8)
train_files = sorted(files[:split_index])
val_files = sorted(files[split_index:])

payload = {
    "seed": SEED,
    "total_images": len(files),
    "train": train_files,
    "validation": val_files,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("Dataset split created successfully.")
print(f"Total images: {len(files)}")
print(f"Training: {len(train_files)}")
print(f"Validation: {len(val_files)}")
print(f"Saved to: {OUTPUT}")
